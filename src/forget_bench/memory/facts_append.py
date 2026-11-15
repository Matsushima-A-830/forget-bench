"""facts_append: (項目, 値, 時刻) を追記のみで保持する事実ストア(Day 028 AppendOnly)。

- 抽出は `extractor` で切り替える。
  - "llm"(既定): ユーザー発話ごとに、回答用と同じバックエンドで事実を抽出する。呼び出しはコストに計上される。
  - "oracle": 生成器の正解ラベル(Annotation)をそのまま使う。抽出が完璧だった場合の参照値。
    正解ラベルを読むので、他の戦略と同列の「戦略」としては扱わない(結果では別扱いにする)。
- 削除要求: 該当レコードに削除フラグを立て、値を消して項目名だけ残す(墓標)。
  対象は、要求の source_turn_ids から抽出されたレコードと、値を含むレコード。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from forget_bench.backends.base import BackendError
from forget_bench.memory.base import StrategyContext, register_strategy
from forget_bench.memory.render import question_text, session_header
from forget_bench.prompts import load_prompt
from forget_bench.scoring.scorer import contains_value
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn

_HEADER = {
    "ja": "以下はこれまでの会話から記録した事実です(古い順。同じ項目が複数あるときは後のものほど新しい)。",
    "en": "Below are facts recorded from the conversation so far (oldest first; for repeated fields, later entries are newer).",
}
_DELETED = {"ja": "(ユーザーの要求により削除済み)", "en": "(deleted at the user's request)"}
_EMPTY = {"ja": "(記録なし)", "en": "(no records)"}
_EXTRACT_MAX_TOKENS = 256


@dataclass
class FactRecord:
    seq: int
    session: int
    key: str
    value: str
    source_turn_id: str
    deleted: bool = False


def parse_facts(text: str) -> list[tuple[str, str]] | None:
    """抽出結果のJSON配列をパースする。失敗したら None。"""
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, dict) and "key" in item and "value" in item:
            key, value = str(item["key"]).strip(), str(item["value"]).strip()
            if key and value:
                out.append((key, value))
    return out


@register_strategy("facts_append")
class FactsAppendMemory:
    name = "facts_append"

    def __init__(self, ctx: StrategyContext, extractor: Literal["llm", "oracle"] = "llm", **_: Any) -> None:
        if extractor not in ("llm", "oracle"):
            raise ValueError("extractor は llm か oracle")
        self.ctx = ctx
        self.extractor = extractor
        self.records: list[FactRecord] = []
        self._extract_prompt = load_prompt("facts_extract", ctx.lang) if extractor == "llm" else ""
        self._llm_calls = 0
        self._errors = 0

    @staticmethod
    def estimate_memory_calls(n_user_turns: int, extractor: str = "llm", **_: Any) -> int:
        return n_user_turns if extractor == "llm" else 0

    def _extract(self, turn: Turn) -> list[tuple[str, str]]:
        if self.extractor == "oracle":
            a = turn.annotation
            return [(a.label, a.value)] if a is not None else []
        self._llm_calls += 1
        try:
            gen = self.ctx.llm.generate(
                [Message("system", self._extract_prompt), Message("user", turn.text)],
                max_tokens=_EXTRACT_MAX_TOKENS,
            )
        except BackendError:
            self._errors += 1
            return []
        facts = parse_facts(gen.text)
        if facts is None:
            self._errors += 1
            return []
        return facts

    def observe(self, turn: Turn) -> None:
        if turn.role != "user":
            return
        for key, value in self._extract(turn):
            self.records.append(FactRecord(len(self.records), turn.session, key, value, turn.id))

    def forget(self, request: ForgetRequest) -> None:
        sources = set(request.source_turn_ids)
        for r in self.records:
            if r.deleted:
                continue
            if r.source_turn_id in sources or any(contains_value(r.value, a) for a in request.aliases):
                r.deleted = True
                r.value = ""

    def build_context(self, question: Question) -> list[Message]:
        lang = self.ctx.lang
        lines = []
        for r in self.records:
            value = _DELETED[lang] if r.deleted else r.value
            lines.append(f"- {session_header(r.session, lang)} {r.key}: {value}")
        body = "\n".join(lines) if lines else _EMPTY[lang]
        return [
            Message("system", self.ctx.system_prompt),
            Message("user", f"{_HEADER[lang]}\n{body}\n\n{question_text(question, lang)}"),
        ]

    def stats(self) -> MemoryStats:
        live = [r for r in self.records if not r.deleted]
        return MemoryStats(
            items=len(live),
            chars=sum(len(r.key) + len(r.value) for r in self.records),
            deleted_items=len(self.records) - len(live),
            llm_calls=self._llm_calls,
            errors=self._errors,
        )
