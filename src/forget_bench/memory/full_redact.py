"""full_redact: 全ターンを保持し、削除要求された値を該当ターンで伏字にする。

- 対象: 削除要求の source_turn_ids のターンと、値(表記揺れ含む)を含むすべてのターン
- 値の表記をそのまま置換し、それでも採点器の判定で値が残る(表記揺れで置換できない)場合はターン全体を伏字にする
- 伏字は「[削除済み: 項目名]」。項目名を残すので、モデルは「削除済み」と答える根拠を持てる
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from forget_bench.memory.base import StrategyContext, register_strategy
from forget_bench.memory.snapshot import turns_snapshot
from forget_bench.memory.render import question_text, render_turns
from forget_bench.scoring.scorer import contains_value
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn

_MASK = {"ja": "[削除済み: {label}]", "en": "[deleted: {label}]"}


@register_strategy("full_redact")
class FullRedactMemory:
    name = "full_redact"

    def __init__(self, ctx: StrategyContext, **_: Any) -> None:
        self.ctx = ctx
        self.turns: list[Turn] = []
        self.redacted_turns: set[str] = set()

    def observe(self, turn: Turn) -> None:
        self.turns.append(turn)

    def forget(self, request: ForgetRequest) -> None:
        mask = _MASK[self.ctx.lang].format(label=request.label)
        sources = set(request.source_turn_ids)
        for i, t in enumerate(self.turns):
            if t.id not in sources and not any(contains_value(t.text, a) for a in request.aliases):
                continue
            text = t.text
            for alias in sorted(request.aliases, key=len, reverse=True):
                text = text.replace(alias, mask)
            if any(contains_value(text, a) for a in request.aliases):
                text = mask
            if text != t.text:
                self.turns[i] = replace(t, text=text)
                self.redacted_turns.add(t.id)

    def build_context(self, question: Question) -> list[Message]:
        return [
            Message("system", self.ctx.system_prompt),
            *render_turns(self.turns, self.ctx.lang),
            Message("user", question_text(question, self.ctx.lang)),
        ]

    def stats(self) -> MemoryStats:
        return MemoryStats(
            items=len(self.turns),
            chars=sum(len(t.text) for t in self.turns),
            deleted_items=len(self.redacted_turns),
        )

    def snapshot(self) -> dict[str, Any]:
        return turns_snapshot(self.turns, self.ctx.lang, redacted=self.redacted_turns)
