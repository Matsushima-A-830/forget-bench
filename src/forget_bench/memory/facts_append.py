"""facts_append: (項目, 値, 時刻) を追記のみで保持する事実ストア(Day 028 AppendOnly)。

- 抽出は `extractor` で切り替える。
  - "llm"(既定): ユーザー発話ごとに、回答用と同じバックエンドで事実を抽出する。呼び出しはコストに計上される。
  - "oracle": 生成器の正解ラベル(Annotation)をそのまま使う。抽出が完璧だった場合の参照値。
    正解ラベルを読むので、他の戦略と同列の「戦略」としては扱わない(結果では別扱いにする)。
- `extract_every`: "turn"(既定)は発話ごとに1回、"session" はセッション内のユーザー発話をまとめて1回抽出する
  (呼び出し数がセッション数になる。claude の利用枠で回せる規模にするため)。
- 削除要求: 該当レコードに削除フラグを立て、値を消して項目名だけ残す(墓標)。
  対象は、要求の source_turn_ids から抽出されたレコードと、値を含むレコード。
  session 抽出ではまだ抽出していない発話があるので、次の抽出の直後に適用する(質問より前に必ず適用される)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from forget_bench.backends.base import BackendError
from forget_bench.memory.base import ScenarioShape, StrategyContext, register_strategy
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
_BATCH_MAX_TOKENS = 1024


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
    items = parse_fact_items(text)
    return None if items is None else [(k, v) for k, v, _ in items]


def _json_values(text: str) -> list[object]:
    """文字列中の JSON の値(配列・オブジェクト)を先頭から順にすべて取り出す。

    モデルは1つの配列のほか、複数の配列を改行で並べたり、オブジェクトを1行ずつ書いたりする。
    """
    dec = json.JSONDecoder()
    out: list[object] = []
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                value, end = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            out.append(value)
            i = end
        else:
            i += 1
    return out


def parse_fact_items(text: str) -> list[tuple[str, str, str | None]] | None:
    """(key, value, turn) のリスト。turn はまとめて抽出したときの発話ID(なければ None)。

    JSON が1つも見つからなければ None(パース失敗)。
    """
    values = _json_values(text)
    if not values:
        return None
    items: list[object] = []
    for v in values:
        items.extend(v if isinstance(v, list) else [v])
    out: list[tuple[str, str, str | None]] = []
    for item in items:
        if isinstance(item, dict) and "key" in item and "value" in item:
            key, value = str(item["key"]).strip(), str(item["value"]).strip()
            turn = str(item["turn"]).strip() if item.get("turn") is not None else None
            if key and value:
                out.append((key, value, turn))
    return out


@register_strategy("facts_append")
class FactsAppendMemory:
    name = "facts_append"

    def __init__(
        self,
        ctx: StrategyContext,
        extractor: Literal["llm", "oracle"] = "llm",
        extract_every: Literal["turn", "session"] = "turn",
        **_: Any,
    ) -> None:
        if extractor not in ("llm", "oracle"):
            raise ValueError("extractor は llm か oracle")
        if extract_every not in ("turn", "session"):
            raise ValueError("extract_every は turn か session")
        self.ctx = ctx
        self.extractor = extractor
        self.extract_every = extract_every
        self.records: list[FactRecord] = []
        self.buffer: list[Turn] = []
        self.pending_forget: list[ForgetRequest] = []
        prompt_name = "facts_extract" if extract_every == "turn" else "facts_extract_batch"
        self._extract_prompt = load_prompt(prompt_name, ctx.lang) if extractor == "llm" else ""
        self._llm_calls = 0
        self._errors = 0

    @staticmethod
    def estimate_memory_calls(
        shape: ScenarioShape, extractor: str = "llm", extract_every: str = "turn", **_: Any
    ) -> int:
        if extractor != "llm":
            return 0
        return shape.n_user_turns if extract_every == "turn" else shape.n_sessions

    # --- 抽出 -----------------------------------------------------------------

    def _llm(self, user: str) -> list[tuple[str, str, str | None]]:
        self._llm_calls += 1
        try:
            gen = self.ctx.llm.generate(
                [Message("system", self._extract_prompt), Message("user", user)],
                max_tokens=_EXTRACT_MAX_TOKENS if self.extract_every == "turn" else _BATCH_MAX_TOKENS,
            )
        except BackendError:
            self._errors += 1
            return []
        items = parse_fact_items(gen.text)
        if items is None:
            self._errors += 1
            return []
        return items

    def _extract_batch(self, turns: list[Turn]) -> None:
        if not turns:
            return
        if self.extractor == "oracle":
            items = [(t.annotation.label, t.annotation.value, t.id) for t in turns if t.annotation is not None]
        elif self.extract_every == "turn":
            (t,) = turns
            items = [(k, v, t.id) for k, v, _ in self._llm(t.text)]
        else:
            items = self._llm("\n".join(f"[{t.id}] {t.text}" for t in turns))
        by_id = {t.id: t for t in turns}
        for key, value, turn_id in items:
            # 1発話ずつなら出所は自明。まとめて抽出したときは LLM が返した発話IDで対応づける
            # (不明なら出所なし。削除要求は値の一致で当てる)
            src = turns[0] if len(turns) == 1 else by_id.get(turn_id or "")
            session = src.session if src else turns[-1].session
            self.records.append(FactRecord(len(self.records), session, key, value, src.id if src else ""))

    def _flush(self) -> None:
        batch, self.buffer = self.buffer, []
        self._extract_batch(batch)
        for req in self.pending_forget:
            self._apply_forget(req)
        self.pending_forget = []

    def observe(self, turn: Turn) -> None:
        if turn.role != "user":
            return
        if self.extract_every == "turn" or self.extractor == "oracle":
            self._extract_batch([turn])
            return
        if self.buffer and turn.session != self.buffer[-1].session:
            self._flush()
        self.buffer.append(turn)

    # --- 削除 -----------------------------------------------------------------

    def _apply_forget(self, request: ForgetRequest) -> None:
        sources = set(request.source_turn_ids)
        for r in self.records:
            if r.deleted:
                continue
            if r.source_turn_id in sources or any(contains_value(r.value, a) for a in request.aliases):
                r.deleted = True
                r.value = ""

    def forget(self, request: ForgetRequest) -> None:
        if self.buffer:
            self.pending_forget.append(request)  # 次の抽出の直後に適用
        else:
            self._apply_forget(request)

    # --- 文脈 -----------------------------------------------------------------

    def facts_block(self) -> str:
        self._flush()
        lang = self.ctx.lang
        lines = []
        for r in self.records:
            value = _DELETED[lang] if r.deleted else r.value
            lines.append(f"- {session_header(r.session, lang)} {r.key}: {value}")
        body = "\n".join(lines) if lines else _EMPTY[lang]
        return f"{_HEADER[lang]}\n{body}"

    def build_context(self, question: Question) -> list[Message]:
        return [
            Message("system", self.ctx.system_prompt),
            Message("user", f"{self.facts_block()}\n\n{question_text(question, self.ctx.lang)}"),
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
