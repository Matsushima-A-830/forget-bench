"""KV 層の戦略: 会話をテキストではなく KV キャッシュとして持ち、忘れるときはキャッシュのトークン範囲を直接切除する。

hf バックエンド(StrategyContext.kv)が必要。

- kv_full: 全ターンを KV に積む。削除要求は無視(テキスト層の full に相当)
- kv_drop: 削除要求された値のターンを KV から切除する。位置補正なし(Day 025 drop。後続のキーの回転と位置が食い違う)
- kv_drop_repos: 切除 + 後続のキーを RoPE 再回転(Day 025 drop+repos)
- kv_evict_old: 直近 max_sessions セッションだけを残す容量制限型。新しいセッションが始まると、最古のセッションを
  丸ごと drop+repos で落とす。削除要求には反応しない(テキスト層の window に相当)

文脈の整形はテキスト層と同じ(システムプロンプト → 発話ごとの ChatML、セッションの最初の発話に見出し)。
1発話を1つのトークン範囲(TokenSpan)として記録する。質問は KV の後ろに続けて生成し、生成後は元の長さに戻す。
"""

from __future__ import annotations

from typing import Any

from forget_bench.memory.base import StrategyContext, register_strategy
from forget_bench.memory.render import question_text, session_header
from forget_bench.scoring.scorer import contains_value
from forget_bench.types import ForgetRequest, Generation, MemoryStats, Message, Question, Turn


class _KVMemory:
    name = "kv"

    def __init__(self, ctx: StrategyContext, **_: Any) -> None:
        if ctx.kv is None:
            raise ValueError(f"{self.name} には KV を操作できるバックエンド(hf)が必要")
        self.ctx = ctx
        self.kv = ctx.kv
        self.handle = self.kv.prefill([Message("system", ctx.system_prompt)])
        self.messages: dict[str, Message] = {}  # 範囲ID → その範囲の発話(ログ・漏洩の確認用)
        self.sessions: dict[str, int] = {}
        self._current_session: int | None = None
        self.peak_vram_mb = 0.0
        self._deleted = 0  # 切除した発話数

    def observe(self, turn: Turn) -> None:
        text = turn.text
        if turn.role == "user" and turn.session != self._current_session:
            text = f"{session_header(turn.session, self.ctx.lang)}\n{text}"
            self._current_session = turn.session
            self._on_new_session(turn.session)
        msg = Message(turn.role, text)
        self.kv.append(self.handle, turn.id, [msg])
        self.messages[turn.id] = msg
        self.sessions[turn.id] = turn.session

    def _on_new_session(self, session: int) -> None:
        pass

    def _drop(self, span_ids: list[str], *, repos: bool) -> None:
        self.kv.drop(self.handle, span_ids, repos=repos)
        for sid in span_ids:
            self.messages.pop(sid, None)
            self.sessions.pop(sid, None)

    def _forget_targets(self, request: ForgetRequest) -> list[str]:
        sources = set(request.source_turn_ids)
        return [
            sid for sid, m in self.messages.items()
            if sid in sources or any(contains_value(m.content, a) for a in request.aliases)
        ]

    def forget(self, request: ForgetRequest) -> None:
        pass

    def _prompt(self, question: Question) -> list[Message]:
        return [Message("user", question_text(question, self.ctx.lang))]

    def answer(self, question: Question) -> Generation:
        gen = self.kv.generate_from(self.handle, self._prompt(question), max_tokens=self.ctx.max_tokens)
        peak = (gen.raw or {}).get("peak_vram_mb")
        if peak:
            self.peak_vram_mb = max(self.peak_vram_mb, float(peak))
        return gen

    def build_context(self, question: Question) -> list[Message]:
        """KV に残っている内容をテキストで表したもの(記録・確認用。回答は answer() が KV から生成する)。"""
        kept = [self.messages[s.id] for s in self.handle.spans if s.id in self.messages]
        return [Message("system", self.ctx.system_prompt), *kept, *self._prompt(question)]

    def stats(self) -> MemoryStats:
        return MemoryStats(
            items=len(self.messages),
            chars=sum(len(m.content) for m in self.messages.values()),
            deleted_items=self._deleted,
            kv_tokens=self.handle.length,
            prefill_tokens=self.handle.prefill_tokens,
            prefill_s=self.handle.prefill_s,
            dropped_tokens=self.handle.dropped_tokens,
            peak_vram_mb=self.peak_vram_mb,
        )


@register_strategy("kv_full")
class KVFull(_KVMemory):
    name = "kv_full"


class _KVDropBase(_KVMemory):
    repos = False

    def forget(self, request: ForgetRequest) -> None:
        targets = self._forget_targets(request)
        if targets:
            self._drop(targets, repos=self.repos)
            self._deleted += len(targets)


@register_strategy("kv_drop")
class KVDrop(_KVDropBase):
    name = "kv_drop"
    repos = False


@register_strategy("kv_drop_repos")
class KVDropRepos(_KVDropBase):
    name = "kv_drop_repos"
    repos = True


@register_strategy("kv_evict_old")
class KVEvictOld(_KVMemory):
    name = "kv_evict_old"

    def __init__(self, ctx: StrategyContext, max_sessions: int = 2, **kw: Any) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions は1以上")
        super().__init__(ctx, **kw)
        self.max_sessions = max_sessions

    def _on_new_session(self, session: int) -> None:
        # これから session が始まる。残すのは session を含めて直近 max_sessions 個
        keep_from = session - self.max_sessions + 1
        old = [sid for sid, s in self.sessions.items() if s < keep_from]
        if old:
            self._drop(old, repos=True)
            self._deleted += len(old)
