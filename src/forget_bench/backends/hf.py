"""Hugging Face transformers バックエンド(KV層)。テキストの generate と、KVキャッシュの直接操作の両方を持つ。

方針(docs/backends.md):
- `attn_implementation="eager"` に固定する。Day 025 で、SDPA/flash 系では値が同じ KV でもメモリレイアウトの違いで
  損失が変わる(恒等変換のはずの delta=0 回転で loss 1.43→2.19)ことを確認したため。eager 以外は明示しないと受け付けない。
- 自由生成(greedy)の回答を採点する。温度は 0 のみ。
- プロンプトは Qwen の ChatML(`<|im_start|>role\\n…<|im_end|>\\n`)で組み立てる。ollama と同じ messages をそのまま整形する。
- prefill は prefill_chunk トークンずつ行う(eager は注意重みを丸ごと作るので、2万トークンを一度に流すとメモリが足りない)。
- KVHandle はキャッシュと「どのトークン範囲がどの発話か(TokenSpan)」を持つ。位置は常に詰めた連番
  (drop の後は、次のトークンをキャッシュ長の位置から続ける。naive drop ではここで焼き込まれた回転との食い違いが生じる)。

torch / transformers は hf extra。import は使うときまで遅らせる(テキスト層だけの環境でもパッケージを読めるように)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from forget_bench.backends.base import BackendError, register_backend
from forget_bench.types import Generation, Message

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def chatml(messages: list[Message], add_generation_prompt: bool = False) -> str:
    text = "".join(f"{IM_START}{m.role}\n{m.content}{IM_END}\n" for m in messages)
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text


@dataclass(frozen=True)
class TokenSpan:
    id: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class KVHandle:
    cache: Any  # transformers.DynamicCache
    spans: list[TokenSpan] = field(default_factory=list)
    prefill_tokens: int = 0  # これまでに prefill したトークン数(コスト)
    prefill_s: float = 0.0
    dropped_tokens: int = 0

    @property
    def length(self) -> int:
        return self.spans[-1].end if self.spans else 0

    def span(self, span_id: str) -> TokenSpan | None:
        return next((s for s in self.spans if s.id == span_id), None)


@register_backend("hf")
class HFBackend:
    name = "hf"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        allow_non_eager: bool = False,
        device: str = "cuda",
        prefill_chunk: int = 512,
        model_obj: Any = None,
        tokenizer_obj: Any = None,
        **_: Any,
    ) -> None:
        import torch

        if attn_implementation != "eager" and not allow_non_eager:
            raise ValueError("attn_implementation は eager に固定(Day 025 の知見)。比較のためだけに allow_non_eager=True で変えられる")
        self.torch = torch
        self.model_name = model
        self.revision = revision
        self.device = device
        self.prefill_chunk = prefill_chunk
        self.attn_implementation = attn_implementation
        if model_obj is not None:
            self.model = model_obj
            self.tokenizer = tokenizer_obj
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = getattr(torch, dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=torch_dtype, attn_implementation=attn_implementation
            ).to(device)
            self.revision = self._resolve_revision(model, revision)
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        self.stop_ids = {i for i in (self._token_id(IM_END), getattr(self.tokenizer, "eos_token_id", None)) if i is not None}

    @staticmethod
    def _resolve_revision(model: str, revision: str | None) -> str | None:
        """読み込んだ重みのコミットハッシュ(meta.json に残す)。transformers 5 では config._commit_hash が
        入らないことがあるので、ローカルのスナップショットのディレクトリ名(= コミットハッシュ)から取る。"""
        try:
            from pathlib import Path

            from huggingface_hub import try_to_load_from_cache

            path = try_to_load_from_cache(model, "config.json", revision=revision)
            return Path(path).parent.name if isinstance(path, str) else revision
        except Exception:  # noqa: BLE001 — ローカルパス指定などでは取れない。その場合は指定値を残す
            return revision

    # --- トークン ---------------------------------------------------------------

    def _token_id(self, token: str) -> int | None:
        conv = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if conv is None:
            return None
        tid = conv(token)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        return None if tid is None or tid == unk else int(tid)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # --- 低レベル ---------------------------------------------------------------

    def new_cache(self) -> Any:
        from transformers import DynamicCache

        return DynamicCache(config=self.model.config)

    def _forward(self, cache: Any, ids: list[int], start: int) -> Any:
        """ids を start の位置から prefill し、最後のトークンのロジットを返す。"""
        torch = self.torch
        logits = None
        with torch.no_grad():
            for i in range(0, len(ids), self.prefill_chunk):
                chunk = ids[i : i + self.prefill_chunk]
                pos = torch.arange(start + i, start + i + len(chunk), device=self.device)
                out = self.model(
                    input_ids=torch.tensor([chunk], device=self.device),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=pos,
                    position_ids=pos[None, :],
                )
                logits = out.logits[:, -1, :]
        return logits

    def _decode(self, cache: Any, prompt_ids: list[int], start: int, max_tokens: int) -> tuple[str, int, float]:
        """prompt_ids を流してから greedy に生成する。(テキスト, 出力トークン数, TTFT秒) を返す。"""
        torch = self.torch
        t0 = time.perf_counter()
        logits = self._forward(cache, prompt_ids, start)
        pos = start + len(prompt_ids)
        out_ids: list[int] = []
        ttft = None
        with torch.no_grad():
            for _ in range(max_tokens):
                next_id = int(logits.argmax(dim=-1).item())
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if next_id in self.stop_ids:
                    break
                out_ids.append(next_id)
                p = torch.tensor([pos], device=self.device)
                logits = self.model(
                    input_ids=torch.tensor([[next_id]], device=self.device),
                    past_key_values=cache, use_cache=True, cache_position=p, position_ids=p[None, :],
                ).logits[:, -1, :]
                pos += 1
        return self.decode(out_ids), len(out_ids), ttft or 0.0

    @staticmethod
    def truncate(cache: Any, length: int) -> None:
        """キャッシュを先頭 length トークンに切り詰める(DynamicCache.crop の引数の意味は版で変わったので使わない)。"""
        for layer in cache.layers:
            layer.keys = layer.keys[:, :, :length, :]
            layer.values = layer.values[:, :, :length, :]

    def _peak_reset(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def _peak_mb(self) -> float | None:
        if self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 2**20
        return None

    # --- 契約: テキストの生成 -----------------------------------------------------

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        if temperature != 0.0:
            raise BackendError("hf バックエンドは greedy(temperature=0)のみ")
        ids = self.encode(chatml(messages, add_generation_prompt=True))
        self._peak_reset()
        t0 = time.perf_counter()
        try:
            text, n_out, ttft = self._decode(self.new_cache(), ids, 0, max_tokens)
        except self.torch.cuda.OutOfMemoryError as e:
            self.torch.cuda.empty_cache()
            raise BackendError(f"CUDA out of memory ({len(ids)} input tokens)") from e
        warnings = ["max_tokens reached"] if n_out >= max_tokens else []
        return Generation(text, len(ids), n_out, time.perf_counter() - t0, ttft,
                          raw={"peak_vram_mb": self._peak_mb(), "context_tokens": len(ids)}, warnings=warnings)

    # --- 契約: KV の操作 ----------------------------------------------------------

    def new_handle(self) -> KVHandle:
        return KVHandle(self.new_cache())

    def append(self, kv: KVHandle, span_id: str, messages: list[Message]) -> TokenSpan:
        """messages を ChatML にしてキャッシュの末尾に prefill し、1つの範囲として記録する。"""
        ids = self.encode(chatml(messages))
        start = kv.length
        t0 = time.perf_counter()
        self._forward(kv.cache, ids, start)
        kv.prefill_s += time.perf_counter() - t0
        kv.prefill_tokens += len(ids)
        span = TokenSpan(span_id, start, start + len(ids))
        kv.spans.append(span)
        if kv.cache.get_seq_length() != span.end:
            raise BackendError("キャッシュ長と範囲の記録が食い違った")
        return span

    def prefill(self, messages: list[Message]) -> KVHandle:
        kv = self.new_handle()
        for i, m in enumerate(messages):
            self.append(kv, f"m{i}", [m])
        return kv

    def drop(self, kv: KVHandle, span_ids: list[str], *, repos: bool) -> KVHandle:
        """指定した範囲をキャッシュから切除する。repos=True なら後続のキーを詰めた位置に再回転する。"""
        from forget_bench.backends.kv_ops import drop_tokens

        targets = set(span_ids)
        if not targets & {s.id for s in kv.spans}:
            return kv
        torch = self.torch
        keep = torch.ones(kv.length, dtype=torch.bool)
        for s in kv.spans:
            if s.id in targets:
                keep[s.start : s.end] = False
        drop_tokens(self.model, kv.cache, keep, repos=repos)
        new_spans: list[TokenSpan] = []
        pos = 0
        for s in kv.spans:
            if s.id in targets:
                kv.dropped_tokens += s.length
                continue
            new_spans.append(TokenSpan(s.id, pos, pos + s.length))
            pos += s.length
        kv.spans = new_spans
        if kv.cache.get_seq_length() != kv.length:
            raise BackendError("drop 後のキャッシュ長と範囲の記録が食い違った")
        return kv

    def generate_from(self, kv: KVHandle, prompt: list[Message], *, max_tokens: int) -> Generation:
        """キャッシュの後ろに prompt を続けて生成する。生成後はキャッシュを元の長さに戻す(質問どうしが影響しない)。"""
        ids = self.encode(chatml(prompt, add_generation_prompt=True))
        base = kv.length
        self._peak_reset()
        t0 = time.perf_counter()
        try:
            text, n_out, ttft = self._decode(kv.cache, ids, base, max_tokens)
        except self.torch.cuda.OutOfMemoryError as e:
            self.torch.cuda.empty_cache()
            self.truncate(kv.cache, base)
            raise BackendError(f"CUDA out of memory ({base} cached + {len(ids)} new tokens)") from e
        self.truncate(kv.cache, base)
        warnings = ["max_tokens reached"] if n_out >= max_tokens else []
        return Generation(text, len(ids), n_out, time.perf_counter() - t0, ttft,
                          raw={"peak_vram_mb": self._peak_mb(), "context_tokens": base + len(ids)}, warnings=warnings)

    def info(self) -> dict[str, Any]:
        import transformers

        cfg = self.model.config
        out: dict[str, Any] = {
            "backend": "hf",
            "model": self.model_name,
            "revision": self.revision or getattr(cfg, "_commit_hash", None),
            "dtype": str(next(self.model.parameters()).dtype),
            "attn_implementation": getattr(cfg, "_attn_implementation", self.attn_implementation),
            "prefill_chunk": self.prefill_chunk,
            "torch": self.torch.__version__,
            "transformers": transformers.__version__,
            "decoding": "greedy",
        }
        if self.device.startswith("cuda"):
            out["gpu"] = self.torch.cuda.get_device_name(0)
        return out
