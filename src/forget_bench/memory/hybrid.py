"""hybrid: facts_append(事実ストア)+ rag(生の発話の検索)を組み合わせる。

- 文脈は「記録した事実」と「検索で選んだ過去の発話」の両方
- 削除要求は両方に適用する(事実は墓標化、発話は索引から削除)
- 事実抽出のパラメータ(extractor / extract_every)と検索のパラメータ(top_k / mode / rrf_k)をそのまま受ける
"""

from __future__ import annotations

from typing import Any

from forget_bench.memory.base import ScenarioShape, StrategyContext, register_strategy
from forget_bench.memory.facts_append import FactsAppendMemory
from forget_bench.memory.rag import _HEADER as RAG_HEADER
from forget_bench.memory.rag import Retriever
from forget_bench.memory.render import question_text
from forget_bench.types import ForgetRequest, MemoryStats, Message, Question, Turn


@register_strategy("hybrid")
class HybridMemory:
    name = "hybrid"

    def __init__(
        self,
        ctx: StrategyContext,
        extractor: str = "llm",
        extract_every: str = "turn",
        top_k: int = 8,
        mode: str = "dense",
        rrf_k: int = 60,
        **_: Any,
    ) -> None:
        self.ctx = ctx
        self.facts = FactsAppendMemory(ctx, extractor=extractor, extract_every=extract_every)  # type: ignore[arg-type]
        self.retriever = Retriever(ctx, top_k, mode, rrf_k)

    @staticmethod
    def estimate_memory_calls(shape: ScenarioShape, **params: Any) -> int:
        keys = ("extractor", "extract_every")
        return FactsAppendMemory.estimate_memory_calls(shape, **{k: v for k, v in params.items() if k in keys})

    def observe(self, turn: Turn) -> None:
        self.facts.observe(turn)
        if turn.role == "user":
            self.retriever.add(turn)

    def forget(self, request: ForgetRequest) -> None:
        self.facts.forget(request)
        self.retriever.remove_for(request)

    def build_context(self, question: Question) -> list[Message]:
        lang = self.ctx.lang
        docs = self.retriever.render(self.retriever.search(question.text))
        body = f"{self.facts.facts_block()}\n\n{RAG_HEADER[lang]}\n{docs}\n\n{question_text(question, lang)}"
        return [Message("system", self.ctx.system_prompt), Message("user", body)]

    def stats(self) -> MemoryStats:
        f = self.facts.stats()
        docs = self.retriever.docs.values()
        return MemoryStats(
            items=f.items + len(docs),
            chars=f.chars + sum(len(d.text) for d in docs),
            deleted_items=f.deleted_items + self.retriever.removed,
            llm_calls=f.llm_calls,
            errors=f.errors,
        )
