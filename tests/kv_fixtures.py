"""KV 層のテスト用: 乱数初期化の小さな Qwen2 と、バイト単位の偽トークナイザ(ダウンロード不要・CPU)。"""

from __future__ import annotations

from typing import Any

SPECIALS = {"<|im_start|>": 256, "<|im_end|>": 257}


class ByteTokenizer:
    eos_token_id = 257
    unk_token_id = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for tok, tid in SPECIALS.items():
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                ids.extend(text[i].encode("utf-8"))
                i += 1
        return {"input_ids": ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return SPECIALS.get(token)


def tiny_model(attn: str = "eager", seed: int = 0) -> Any:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        rope_theta=10000.0, attn_implementation=attn,
    )
    return Qwen2ForCausalLM(cfg).to(torch.float32).eval()


def tiny_backend(prefill_chunk: int = 16, attn: str = "eager") -> Any:
    from forget_bench.backends.hf import HFBackend

    return HFBackend("tiny-random-qwen2", model_obj=tiny_model(attn), tokenizer_obj=ByteTokenizer(),
                     device="cpu", prefill_chunk=prefill_chunk, attn_implementation=attn,
                     allow_non_eager=attn != "eager")
