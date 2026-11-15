from __future__ import annotations

from forget_bench.backends import make_backend
from forget_bench.memory import StrategyContext, make_strategy
from forget_bench.memory.facts_append import parse_facts
from forget_bench.scenario import generate
from forget_bench.scoring import contains_value
from forget_bench.types import Generation, Message, Scenario
from tests.conftest import make_params


def _ctx(llm=None) -> StrategyContext:  # noqa: ANN001
    return StrategyContext("ja", "SYSTEM", llm or make_backend("dummy"), 256)


def _feed(strategy, sc: Scenario) -> None:  # noqa: ANN001
    for t in sc.turns:
        strategy.observe(t)
        req = sc.forget_request_for_turn(t.id)
        if req:
            strategy.forget(req)


def _context_text(strategy, sc: Scenario) -> str:  # noqa: ANN001
    msgs = strategy.build_context(sc.questions[0])
    assert msgs[0] == Message("system", "SYSTEM")
    assert msgs[-1].role == "user" and sc.questions[0].text in msgs[-1].content
    return "\n".join(m.content for m in msgs)


def _leaks(text: str, sc: Scenario) -> bool:
    return any(contains_value(text, a) for r in sc.forget_requests for a in r.aliases)


def test_full_keeps_everything_and_ignores_forget() -> None:
    sc = generate(1000, make_params())
    s = make_strategy("full", _ctx())
    _feed(s, sc)
    text = _context_text(s, sc)
    assert all(t.text in text for t in sc.turns)
    assert _leaks(text, sc)
    assert "[セッション 1]" in text and "[セッション 5]" in text


def test_window_keeps_last_n_user_turns() -> None:
    sc = generate(1000, make_params())
    s = make_strategy("window", _ctx(), n_turns=4)
    _feed(s, sc)
    msgs = s.build_context(sc.questions[0])[1:-1]
    assert [m.role for m in msgs] == ["user", "assistant"] * 4
    last_users = [t for t in sc.turns if t.role == "user"][-4:]
    assert last_users[0].text in msgs[0].content
    assert s.stats().items == 8


def test_facts_oracle_forgets_with_tombstone() -> None:
    sc = generate(1000, make_params())
    s = make_strategy("facts_append", _ctx(), extractor="oracle")
    _feed(s, sc)
    text = _context_text(s, sc)
    assert not _leaks(text, sc)
    assert "削除済み" in text
    st = s.stats()
    assert st.deleted_items == len(sc.forget_requests) and st.llm_calls == 0
    # 削除していない値は残っている
    kept = [t.annotation.value for t in sc.turns if t.annotation and t.annotation.fact_key == "aggregate"]
    assert all(v in text for v in kept)


class _ScriptedLLM:
    """抽出用: 発話中の「は」以降を値として返す偽LLM。呼び出しを数える。"""

    name = "scripted"

    def __init__(self, sc: Scenario) -> None:
        self.calls = 0
        self.by_text = {t.text: t.annotation for t in sc.turns if t.role == "user"}

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        self.calls += 1
        a = self.by_text.get(messages[-1].content)
        body = f'```json\n[{{"key": "{a.label}", "value": "{a.value}"}}]\n```' if a else "[]"
        return Generation(body, 10, 5, 0.0, None)

    def info(self) -> dict[str, str]:
        return {}


def test_facts_llm_extraction_and_forget() -> None:
    sc = generate(1001, make_params())
    llm = _ScriptedLLM(sc)
    s = make_strategy("facts_append", _ctx(llm))
    _feed(s, sc)
    n_user = sum(t.role == "user" for t in sc.turns)
    assert llm.calls == n_user == s.stats().llm_calls
    assert not _leaks(_context_text(s, sc), sc)


def test_parse_facts() -> None:
    assert parse_facts('[{"key": "都市", "value": "金沢"}]') == [("都市", "金沢")]
    assert parse_facts('前置き\n[{"key":"a","value":1}] 後書き') == [("a", "1")]
    assert parse_facts("[]") == []
    assert parse_facts("no json") is None
    assert parse_facts("[{broken") is None
