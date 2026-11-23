"""rag: ユーザー発話をそのまま索引に入れ、質問ごとに検索した上位 top_k 件を文脈にする(Day 028 dense+rerank)。

- 検索: mode="dense"(既定)は埋め込みのコサイン類似度、"bm25" は BM25、
  "fusion" は両者の順位を RRF で融合する(Day 028 の dense+rerank の rerank 部分の代わり)。
  既定を dense にしたのは dev 分割での検索再現率による(2026-11-15、bge-m3、top_k=8、current+oneshot の正解を含む率):
  日本語で dense 24/24・fusion 21/24・bm25 19/24(5セッション)。文字バイグラムの BM25 が日本語で弱く、融合で下がった。
- 取り出した発話は時系列順に並べて渡す(変化した属性の新旧をモデルが判断できるように)。
- 削除要求: 該当ターン(source_turn_ids と、値を含むターン)を索引から消す。
  「忘れて」という発話そのものは索引に残るので、モデルは「削除済み」と答える根拠を検索で得られる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from forget_bench.memory.base import StrategyContext, register_strategy
from forget_bench.memory.render import question_text, session_header
from forget_bench.retrieval import BM25Index, rrf
from forget_bench.scoring.scorer import contains_value
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn

_HEADER = {
    "ja": "以下は、質問に関係しそうな過去のユーザーの発話です(検索で選んだもの。古い順)。",
    "en": "Below are past user utterances that may be relevant to the question (selected by search; oldest first).",
}
_EMPTY = {"ja": "(該当なし)", "en": "(none)"}


@dataclass(frozen=True)
class Doc:
    id: str
    index: int
    session: int
    text: str


class Retriever:
    """rag と hybrid が共有する検索器。ユーザー発話を文書として持つ。"""

    def __init__(self, ctx: StrategyContext, top_k: int, mode: str, rrf_k: int) -> None:
        if mode not in ("dense", "bm25", "fusion"):
            raise ValueError("mode は dense / bm25 / fusion")
        if mode != "bm25" and ctx.embedder is None:
            raise ValueError("密検索には埋め込み(StrategyContext.embedder)が必要")
        self.ctx = ctx
        self.top_k = top_k
        self.mode = mode
        self.rrf_k = rrf_k
        self.docs: dict[str, Doc] = {}
        self.bm25 = BM25Index()
        self.removed = 0

    def add(self, turn: Turn) -> None:
        self.docs[turn.id] = Doc(turn.id, turn.index, turn.session, turn.text)
        self.bm25.add(turn.id, turn.text)

    def remove_for(self, request: ForgetRequest) -> None:
        sources = set(request.source_turn_ids)
        for doc in list(self.docs.values()):
            if doc.id in sources or any(contains_value(doc.text, a) for a in request.aliases):
                del self.docs[doc.id]
                self.bm25.remove(doc.id)
                self.removed += 1

    def snapshot(self) -> dict[str, Any]:
        return {"kind": "docs", "removed": self.removed, "mode": self.mode, "top_k": self.top_k, "items": [
            {"id": d.id, "session": d.session, "text": d.text} for d in sorted(self.docs.values(), key=lambda d: d.index)]}

    def search(self, query: str) -> list[Doc]:
        if not self.docs:
            return []
        ids = list(self.docs)
        rankings: list[list[str]] = []
        if self.mode in ("dense", "fusion"):
            assert self.ctx.embedder is not None
            vecs = self.ctx.embedder.embed([self.docs[i].text for i in ids] + [query])
            qv = vecs[-1]
            sims = {i: sum(a * b for a, b in zip(v, qv)) for i, v in zip(ids, vecs[:-1])}
            rankings.append(sorted(ids, key=lambda i: (-sims[i], self.docs[i].index)))
        if self.mode in ("bm25", "fusion"):
            scores = self.bm25.scores(query)
            rankings.append(sorted(ids, key=lambda i: (-scores.get(i, 0.0), self.docs[i].index)))
        top = rrf(rankings, k=self.rrf_k)[: self.top_k] if len(rankings) > 1 else rankings[0][: self.top_k]
        return sorted((self.docs[i] for i in top), key=lambda d: d.index)

    def render(self, docs: list[Doc]) -> str:
        lang = self.ctx.lang
        if not docs:
            return _EMPTY[lang]
        return "\n".join(f"- {session_header(d.session, lang)} {d.text}" for d in docs)


@register_strategy("rag")
class RagMemory:
    name = "rag"

    def __init__(
        self,
        ctx: StrategyContext,
        top_k: int = 8,
        mode: Literal["dense", "bm25", "fusion"] = "dense",
        rrf_k: int = 60,
        **_: Any,
    ) -> None:
        self.ctx = ctx
        self.retriever = Retriever(ctx, top_k, mode, rrf_k)

    def observe(self, turn: Turn) -> None:
        if turn.role == "user":
            self.retriever.add(turn)

    def forget(self, request: ForgetRequest) -> None:
        self.retriever.remove_for(request)

    def build_context(self, question: Question) -> list[Message]:
        lang = self.ctx.lang
        body = self.retriever.render(self.retriever.search(question.text))
        return [
            Message("system", self.ctx.system_prompt),
            Message("user", f"{_HEADER[lang]}\n{body}\n\n{question_text(question, lang)}"),
        ]

    def stats(self) -> MemoryStats:
        docs = self.retriever.docs.values()
        return MemoryStats(items=len(docs), chars=sum(len(d.text) for d in docs), deleted_items=self.retriever.removed)

    def snapshot(self) -> dict[str, Any]:
        return self.retriever.snapshot()
