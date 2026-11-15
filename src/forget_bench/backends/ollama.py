"""ローカル Ollama バックエンド(HTTP API /api/chat)。

注意(docs/backends.md):
- `num_ctx` を必ず明示する。超えた分は **エラーにならず黙って切り詰められる**
  (Ollama 0.35.1 で確認: 3236トークンの入力が num_ctx=512 で prompt_eval_count=258 になった)。
  そのため、入力文字数から見て prompt_eval_count が少なすぎる場合や、
  入力+出力が num_ctx を超えうる場合は切り詰めとみなしてエラーにする。
- `prompt_eval_count` はプレフィックスキャッシュが効いても全入力トークン数を返す(同上で確認)。
  キャッシュが効くと `prompt_eval_duration` だけが短くなる。
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from forget_bench.backends.base import BackendError, register_backend
from forget_bench.types import Generation, Message

# 1トークンあたりの文字数の上限の目安。英語でも概ね4文字/トークン以下なので、
# prompt_eval_count が「文字数/4」を大きく下回ったら切り詰めを疑う。
_MAX_CHARS_PER_TOKEN = 4.0


@register_backend("ollama")
class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        model: str,
        num_ctx: int,
        *,
        host: str = "http://localhost:11434",
        seed: int = 0,
        timeout_s: float = 600.0,
        keep_alive: str = "30m",
        client: httpx.Client | None = None,
        **_: Any,
    ) -> None:
        if num_ctx <= 0:
            raise ValueError("num_ctx を明示すること(既定値だと黙って切り詰められる)")
        self.model = model
        self.num_ctx = num_ctx
        self.host = host.rstrip("/")
        self.seed = seed
        self.keep_alive = keep_alive
        self._client = client or httpx.Client(timeout=timeout_s)

    # --- 契約 ---------------------------------------------------------------

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        t0 = time.perf_counter()
        try:
            resp = self._client.post(f"{self.host}/api/chat", json=body)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise BackendError(f"ollama request failed: {e}") from e
        latency = time.perf_counter() - t0

        prompt_tokens = data.get("prompt_eval_count")
        out_tokens = data.get("eval_count")
        ns = 1e-9
        ttft = None
        if data.get("prompt_eval_duration") is not None:
            ttft = (data.get("load_duration") or 0) * ns + data["prompt_eval_duration"] * ns
        raw = {
            k: data.get(k)
            for k in (
                "prompt_eval_count", "eval_count", "done_reason", "total_duration",
                "load_duration", "prompt_eval_duration", "eval_duration",
            )
        }
        text = (data.get("message") or {}).get("content", "")

        input_chars = sum(len(m.content) for m in messages)
        if prompt_tokens is not None:
            if prompt_tokens < input_chars / _MAX_CHARS_PER_TOKEN * 0.9:
                raise BackendError(
                    f"context truncation suspected: prompt_eval_count={prompt_tokens} "
                    f"for {input_chars} chars (num_ctx={self.num_ctx})"
                )
            if prompt_tokens + max_tokens > self.num_ctx:
                raise BackendError(
                    f"context overflow: prompt_eval_count={prompt_tokens} + max_tokens={max_tokens} "
                    f"> num_ctx={self.num_ctx}"
                )
        warnings = []
        if data.get("done_reason") == "length":
            warnings.append("max_tokens reached")
        return Generation(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=out_tokens,
            latency_s=latency,
            ttft_s=ttft,
            raw=raw,
            warnings=warnings,
        )

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": "ollama", "model": self.model, "num_ctx": self.num_ctx, "seed": self.seed}
        try:
            out["ollama_version"] = self._client.get(f"{self.host}/api/version").json().get("version")
            tags = self._client.get(f"{self.host}/api/tags").json().get("models", [])
            for m in tags:
                if m.get("name") == self.model or m.get("model") == self.model:
                    out["digest"] = m.get("digest")
                    out["details"] = m.get("details")
            show = self._client.post(f"{self.host}/api/show", json={"model": self.model}).json()
            out["template_head"] = (show.get("template") or "")[:200]
            out["parameters"] = show.get("parameters")
        except (httpx.HTTPError, ValueError) as e:
            out["info_error"] = str(e)
        return out

    def warmup(self) -> None:
        """モデルをロードしておく(最初の質問のTTFTにロード時間を混ぜないため)。"""
        self.generate([Message("user", "ok")], max_tokens=1)
