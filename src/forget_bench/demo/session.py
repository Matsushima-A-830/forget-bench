"""デモ UI の中身: 1つの会話を複数のメモリ戦略に同時に見せ、メモリの状態を JSON で返す(docs/design.md「5. デモUI」)。

- 会話: ユーザーの発話に、主戦略(primary)のメモリで回答する。発話と回答はすべての戦略に observe させる
- 「忘れて」: 発話が削除の指示(下の detect_forget)なら、全戦略に ForgetRequest を渡し、回答は固定文にする
  (LLM に返事を書かせると、消したはずの値を復唱することがあるため)。削除要求の検出に LLM は使わない
- 比較: 同じ質問を全戦略に聞き、消した値が回答に出たか(漏洩)を採点器と同じ規則で判定する。会話には加えない
- シナリオの読み込み: dev 分割のシナリオを流し込む(ランナーと同じ順で observe / forget)。test 分割は読まない

ベンチマークの記録(runs/)は作らない。デモの数値を結果として使わない。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from forget_bench.backends.base import BackendError
from forget_bench.memory import StrategyContext, make_strategy
from forget_bench.prompts import load_prompt
from forget_bench.scenario import generate, split_of
from forget_bench.scoring.scorer import contains_value
from forget_bench.types import ForgetRequest, Lang, Question, ScenarioParams, Turn

_FORGET_WORDS = ("忘れて", "消して", "削除して", "forget", "delete", "remove", "erase")
_QUOTED = re.compile(r"「([^」]+)」|『([^』]+)』|\"([^\"]+)\"|“([^”]+)”|'([^']+)'")
_ACK = {"ja": "わかりました。その情報は記録から削除しました。", "en": "Understood. I have deleted that from my records."}
_LABEL = {"ja": "ユーザーが削除を求めた情報", "en": "information the user asked to delete"}


def detect_forget(text: str) -> list[str]:
    """削除の指示なら、消す値(かぎ括弧や引用符で囲んだ部分)を返す。指示でなければ空。

    例: 「090-1234-5678」を忘れて / Please forget "ABC-123"。値を括弧で囲むのはデモの約束事。
    """
    lowered = text.lower()
    if not any(w in lowered for w in _FORGET_WORDS):
        return []
    values = [next(g for g in m.groups() if g) for m in _QUOTED.finditer(text)]
    return [v.strip() for v in values if v.strip()]


@dataclass
class StrategySpec:
    name: str
    label: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass
class ChatItem:
    role: str  # "user" | "assistant" | "event"
    text: str
    turn_id: str | None = None
    session: int | None = None
    strategy: str | None = None  # 回答を作った戦略
    latency_s: float | None = None
    forget: bool = False


class DemoSession:
    def __init__(self, backend: Any, specs: list[StrategySpec], *, lang: Lang = "ja", primary: str | None = None,
                 embedder: Any = None, max_tokens: int = 256) -> None:
        if not specs:
            raise ValueError("戦略を1つ以上指定する")
        self.backend = backend
        self.specs = specs
        self.lang: Lang = lang
        self.embedder = embedder
        self.max_tokens = max_tokens
        names = [s.display for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"戦略の label が重複している: {names}")
        self.primary = primary or names[0]
        if self.primary not in names:
            raise ValueError(f"primary {self.primary} が戦略にない")
        self.reset()

    # --- 状態 -------------------------------------------------------------------

    def reset(self) -> None:
        kv = self.backend if callable(getattr(self.backend, "generate_from", None)) else None
        ctx = StrategyContext(lang=self.lang, system_prompt=load_prompt("system", self.lang), llm=self.backend,
                              max_tokens=self.max_tokens, embedder=self.embedder, kv=kv)
        self.strategies = {s.display: make_strategy(s.name, ctx, **s.params) for s in self.specs}
        self.chat_log: list[ChatItem] = []
        self.forgotten: list[ForgetRequest] = []
        self.turns: list[Turn] = []
        self.session = 1
        self.index = 0
        self.questions: list[dict[str, Any]] = []
        self.scenario_id: str | None = None

    def new_session(self) -> None:
        if any(t.session == self.session for t in self.turns):
            self.session += 1
            self.chat_log.append(ChatItem("event", f"セッション {self.session} を開始" if self.lang == "ja"
                                          else f"Session {self.session} started"))

    def _observe(self, turn: Turn) -> None:
        self.turns.append(turn)
        for st in self.strategies.values():
            st.observe(turn)

    def _forget(self, req: ForgetRequest) -> None:
        self.forgotten.append(req)
        for st in self.strategies.values():
            st.forget(req)

    def _answer(self, name: str, question: Question) -> tuple[str, float]:
        st = self.strategies[name]
        t0 = time.perf_counter()
        own = getattr(st, "answer", None)
        try:
            if callable(own):
                gen = own(question)
            else:
                gen = self.backend.generate(st.build_context(question), max_tokens=self.max_tokens)
            text = gen.text
        except BackendError as e:
            text = f"[エラー] {e}"
        return text.strip(), time.perf_counter() - t0

    # --- 操作 -------------------------------------------------------------------

    def chat(self, text: str) -> ChatItem:
        text = text.strip()
        if not text:
            raise ValueError("空の発話")
        uid, aid = f"t{self.index:03d}u", f"t{self.index:03d}a"
        values = detect_forget(text)
        if values:
            user = Turn(uid, self.session, self.index, "user", text, "forget")
            self._observe(user)
            for v in values:
                sources = tuple(t.id for t in self.turns if contains_value(t.text, v))
                self._forget(ForgetRequest(f"demo-f{len(self.forgotten)}", "demo", _LABEL[self.lang], v, (v,),
                                           sources, uid))
            reply, latency, by = _ACK[self.lang], 0.0, None
        else:
            q = Question(f"chat-{self.index}", "current", text, None, ())
            reply, latency = self._answer(self.primary, q)
            by = self.primary
            self._observe(Turn(uid, self.session, self.index, "user", text, "noise"))
        self._observe(Turn(aid, self.session, self.index, "assistant", reply, "reply"))
        self.index += 1
        self.chat_log.append(ChatItem("user", text, uid, self.session, forget=bool(values)))
        item = ChatItem("assistant", reply, aid, self.session, strategy=by, latency_s=latency, forget=bool(values))
        self.chat_log.append(item)
        return item

    def forget_value(self, value: str) -> None:
        """UI のボタンから直接消す(発話を介さない明示シグナル)。"""
        value = value.strip()
        if not value:
            raise ValueError("空の値")
        sources = tuple(t.id for t in self.turns if contains_value(t.text, value))
        self._forget(ForgetRequest(f"demo-f{len(self.forgotten)}", "demo", _LABEL[self.lang], value, (value,),
                                   sources, sources[-1] if sources else ""))
        self.chat_log.append(ChatItem("event", f"「{value}」の削除を全戦略に送った" if self.lang == "ja"
                                      else f'Sent deletion of "{value}" to all strategies', forget=True))

    def ask_all(self, text: str) -> dict[str, dict[str, Any]]:
        """同じ質問を全戦略に聞く(会話には加えない)。消した値が回答に出たら leaked。"""
        q = Question(f"ask-{len(self.chat_log)}", "current", text.strip(), None, ())
        out: dict[str, dict[str, Any]] = {}
        for name in self.strategies:
            answer, latency = self._answer(name, q)
            leaked = [r.value for r in self.forgotten if any(contains_value(answer, a) for a in r.aliases)]
            out[name] = {"answer": answer, "latency_s": latency, "leaked": leaked}
        return out

    def load_scenario(self, seed: int, sessions: int = 3, turns: tuple[int, int] = (6, 10)) -> None:
        """dev 分割のシナリオを会話として流し込む(ランナーと同じく、削除要求の発話の直後に forget)。"""
        if split_of(seed) != "dev":
            raise ValueError("デモで読み込むのは dev 分割の seed (1000-9999) だけ")
        params = ScenarioParams(lang=self.lang, sessions=sessions, turns_per_session=turns, n_attributes=4,
                                max_changes=3, n_oneshot=3, n_sensitive=3, n_forget=2, aggregate_mentions=0)
        sc = generate(seed, params)
        self.reset()
        self.scenario_id = sc.id
        for turn in sc.turns:
            self._observe(turn)
            req = sc.forget_request_for_turn(turn.id)
            if turn.role == "user" and turn.session != self.session:
                self.session = turn.session
            self.chat_log.append(ChatItem(turn.role, turn.text, turn.id, turn.session, forget=req is not None))
            if req is not None:
                self._forget(req)
        self.session = sc.params.sessions
        self.index = max(t.index for t in sc.turns) + 1
        self.questions = [{"id": q.id, "category": q.category, "text": q.text, "gold": list(q.gold)}
                          for q in sc.questions]

    # --- 表示 -------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        memory = {}
        for name, st in self.strategies.items():
            snap = getattr(st, "snapshot", None)
            memory[name] = {
                "strategy": next(s.name for s in self.specs if s.display == name),
                "snapshot": snap() if callable(snap) else {"kind": "none"},
                "stats": asdict(st.stats()),
            }
        info = getattr(self.backend, "info", None)
        return {
            "lang": self.lang, "primary": self.primary, "session": self.session, "scenario_id": self.scenario_id,
            "backend": info() if callable(info) else {"backend": getattr(self.backend, "name", "?")},
            "chat": [asdict(c) for c in self.chat_log],
            "forgotten": [{"value": r.value, "turns": list(r.source_turn_ids)} for r in self.forgotten],
            "memory": memory, "questions": self.questions,
        }
