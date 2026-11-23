"""window: 直近 n_turns 回分のユーザー発話(と、その応答)だけを保持する。

削除要求には何もしない。古いものから自然に落ちる。
"""

from __future__ import annotations

from typing import Any

from forget_bench.memory.base import StrategyContext, register_strategy
from forget_bench.memory.snapshot import turns_snapshot
from forget_bench.memory.render import question_text, render_turns
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn


@register_strategy("window")
class WindowMemory:
    name = "window"

    def __init__(self, ctx: StrategyContext, n_turns: int = 10, **_: Any) -> None:
        if n_turns < 1:
            raise ValueError("n_turns は1以上")
        self.ctx = ctx
        self.n_turns = n_turns
        self.turns: list[Turn] = []

    def observe(self, turn: Turn) -> None:
        self.turns.append(turn)
        indices = sorted({t.index for t in self.turns})
        if len(indices) > self.n_turns:
            keep_from = indices[-self.n_turns]
            self.turns = [t for t in self.turns if t.index >= keep_from]

    def forget(self, request: ForgetRequest) -> None:
        pass

    def build_context(self, question: Question) -> list[Message]:
        return [
            Message("system", self.ctx.system_prompt),
            *render_turns(self.turns, self.ctx.lang),
            Message("user", question_text(question, self.ctx.lang)),
        ]

    def stats(self) -> MemoryStats:
        return MemoryStats(items=len(self.turns), chars=sum(len(t.text) for t in self.turns))

    def snapshot(self) -> dict[str, Any]:
        return turns_snapshot(self.turns, self.ctx.lang, note=f"直近 {self.n_turns} 発話")
