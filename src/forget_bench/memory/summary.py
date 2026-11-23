"""summary: LLM が更新し続ける要約1本だけを保持する(Day 028 Mutating / rolling summary)。

- update_every="session"(既定): セッションの区切りで、溜めた会話を要約に反映する。"turn" なら発話ごと
- 削除要求: まず溜めている会話を反映し、続けて要約更新プロンプトで「その値を消して」と指示する
- 要約の更新は回答用と同じバックエンドで行い、その呼び出しもコストに計上される
- 更新に失敗したら要約は前のまま。溜めた会話は捨てずに次の更新に回す
"""

from __future__ import annotations

from typing import Any, Literal

from forget_bench.backends.base import BackendError
from forget_bench.memory.base import ScenarioShape, StrategyContext, register_strategy
from forget_bench.memory.render import question_text, session_header
from forget_bench.prompts import load_prompt
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn

_UPDATE_MAX_TOKENS = 1024
_TEXT = {
    "ja": {
        "current": "現在の要約:",
        "empty": "(まだない)",
        "new": "新しい会話:",
        "user": "ユーザー",
        "assistant": "アシスタント",
        "delete": "削除要求: ユーザーが「{label}」({value})を記録から消すよう求めた。",
        "tail": "更新した要約だけを出力してください。",
        "context": "これまでの会話の要約:",
    },
    "en": {
        "current": "Current summary:",
        "empty": "(none yet)",
        "new": "New conversation:",
        "user": "User",
        "assistant": "Assistant",
        "delete": 'Deletion request: the user asked to delete "{label}" ({value}) from the records.',
        "tail": "Output only the updated summary.",
        "context": "Summary of the conversation so far:",
    },
}


@register_strategy("summary")
class SummaryMemory:
    name = "summary"

    def __init__(self, ctx: StrategyContext, update_every: Literal["session", "turn"] = "session", **_: Any) -> None:
        if update_every not in ("session", "turn"):
            raise ValueError("update_every は session か turn")
        self.ctx = ctx
        self.update_every = update_every
        self.summary = ""
        self.buffer: list[Turn] = []
        self._prompt = load_prompt("summary_update", ctx.lang)
        self._llm_calls = 0
        self._errors = 0
        self._deletions = 0

    @staticmethod
    def estimate_memory_calls(shape: ScenarioShape, update_every: str = "session", **_: Any) -> int:
        # 区切りごとに1回 + 削除要求ごとに「途中までの反映」と「削除」で2回
        units = shape.n_sessions if update_every == "session" else shape.n_user_turns
        return units + 2 * shape.n_forget

    def _t(self, key: str) -> str:
        return _TEXT[self.ctx.lang][key]

    def _update(self, body: str) -> bool:
        current = self.summary or self._t("empty")
        user = f"{self._t('current')}\n{current}\n\n{body}\n\n{self._t('tail')}"
        self._llm_calls += 1
        try:
            gen = self.ctx.llm.generate(
                [Message("system", self._prompt), Message("user", user)], max_tokens=_UPDATE_MAX_TOKENS
            )
        except BackendError:
            self._errors += 1
            return False
        text = gen.text.strip()
        if not text:
            self._errors += 1
            return False
        self.summary = text
        return True

    def _flush(self) -> None:
        if not self.buffer:
            return
        lines = [self._t("new")]
        current: int | None = None
        for t in self.buffer:
            if t.session != current:
                lines.append(session_header(t.session, self.ctx.lang))
                current = t.session
            lines.append(f"{self._t(t.role)}: {t.text}")
        if self._update("\n".join(lines)):
            self.buffer = []

    def observe(self, turn: Turn) -> None:
        if self.buffer:
            boundary = (
                turn.session != self.buffer[-1].session
                if self.update_every == "session"
                else turn.role == "user"
            )
            if boundary:
                self._flush()
        self.buffer.append(turn)

    def forget(self, request: ForgetRequest) -> None:
        self._flush()
        self._deletions += 1
        self._update(self._t("delete").format(label=request.label, value=request.value))

    def build_context(self, question: Question) -> list[Message]:
        self._flush()
        lang = self.ctx.lang
        body = self.summary or self._t("empty")
        return [
            Message("system", self.ctx.system_prompt),
            Message("user", f"{self._t('context')}\n{body}\n\n{question_text(question, lang)}"),
        ]

    def stats(self) -> MemoryStats:
        return MemoryStats(
            items=1 if self.summary else 0,
            chars=len(self.summary),
            deleted_items=self._deletions,
            llm_calls=self._llm_calls,
            errors=self._errors,
        )

    def snapshot(self) -> dict[str, Any]:
        return {"kind": "text", "text": self.summary, "pending_turns": len(self.buffer),
                "updates": self._llm_calls, "deletions": self._deletions}
