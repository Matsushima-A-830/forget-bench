"""full: 全ターンを保持し、削除要求は無視する(漏洩の上限側の参照値)。"""

from __future__ import annotations

from typing import Any

from forget_bench.memory.base import StrategyContext, register_strategy
from forget_bench.memory.snapshot import turns_snapshot
from forget_bench.memory.render import question_text, render_turns
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn


@register_strategy("full")
class FullMemory:
    name = "full"

    def __init__(self, ctx: StrategyContext, **_: Any) -> None:
        self.ctx = ctx
        self.turns: list[Turn] = []

    def observe(self, turn: Turn) -> None:
        self.turns.append(turn)

    def forget(self, request: ForgetRequest) -> None:
        pass  # 意図的に何もしない

    def build_context(self, question: Question) -> list[Message]:
        return [
            Message("system", self.ctx.system_prompt),
            *render_turns(self.turns, self.ctx.lang),
            Message("user", question_text(question, self.ctx.lang)),
        ]

    def stats(self) -> MemoryStats:
        return MemoryStats(items=len(self.turns), chars=sum(len(t.text) for t in self.turns))

    def snapshot(self) -> dict[str, Any]:
        return turns_snapshot(self.turns, self.ctx.lang)
