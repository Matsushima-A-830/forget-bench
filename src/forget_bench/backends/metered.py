"""バックエンド呼び出しを1リクエスト1行で記録するラッパー。

メモリ戦略にはこのラッパーを purpose="memory" で渡すので、
要約・抽出などメモリ構築のための呼び出しも自動的にコストに計上される(AGENTS.md)。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from forget_bench.backends.base import Backend, BackendError
from forget_bench.scoring.summary import RequestRow
from forget_bench.types import Generation, Message


@dataclass(frozen=True)
class RequestTags:
    lang: str
    scenario_id: str
    strategy: str
    purpose: str
    question_id: str | None = None


class MeteredBackend:
    def __init__(self, inner: Backend, sink: Callable[[RequestRow], None], tags: RequestTags) -> None:
        self.inner = inner
        self.sink = sink
        self.tags = tags
        self.name = inner.name

    def with_tags(self, **changes: Any) -> "MeteredBackend":
        return MeteredBackend(self.inner, self.sink, replace(self.tags, **changes))

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        row: dict[str, Any] = {
            "lang": self.tags.lang,
            "scenario_id": self.tags.scenario_id,
            "strategy": self.tags.strategy,
            "purpose": self.tags.purpose,
            "question_id": self.tags.question_id,
            "n_messages": len(messages),
            "input_chars": sum(len(m.content) for m in messages),
        }
        try:
            gen = self.inner.generate(messages, max_tokens=max_tokens, temperature=temperature)
        except BackendError as e:
            self.sink(RequestRow(**row, input_tokens=None, output_tokens=None, latency_s=None,
                                 ttft_s=None, output=None, error=str(e), warnings=[]))
            raise
        self.sink(RequestRow(**row, input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
                             latency_s=gen.latency_s, ttft_s=gen.ttft_s, output=gen.text,
                             error=None, warnings=list(gen.warnings)))
        return gen

    def info(self) -> dict[str, Any]:
        return self.inner.info()
