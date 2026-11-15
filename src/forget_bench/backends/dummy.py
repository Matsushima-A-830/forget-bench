"""LLMなしで全経路を通すための決定的なバックエンド(テスト・ドライラン用)。

- mode="constant": 常に同じ文字列を返す
- mode="echo": 渡された文脈(system 以外)をそのまま返す。
  文脈に値が残っていれば必ず「漏洩」になるので、メモリ戦略の忘れ方の配管テストに使える。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from forget_bench.backends.base import register_backend
from forget_bench.types import Generation, Message


@register_backend("dummy")
class DummyBackend:
    name = "dummy"

    def __init__(self, mode: Literal["constant", "echo"] = "constant", text: str = "不明", **_: Any) -> None:
        self.mode = mode
        self.text = text

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        t0 = time.perf_counter()
        body = "\n".join(m.content for m in messages if m.role != "system")
        out = body if self.mode == "echo" else self.text
        return Generation(
            text=out,
            input_tokens=None,  # トークン数は数えない(文字数は requests.jsonl の input_chars にある)
            output_tokens=None,
            latency_s=time.perf_counter() - t0,
            ttft_s=None,
        )

    def info(self) -> dict[str, Any]:
        return {"backend": "dummy", "mode": self.mode}
