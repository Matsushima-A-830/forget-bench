from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from forget_bench.scenario import generate, seeds_for, split_of
from forget_bench.scoring import contains_value
from forget_bench.types import LEAK_CATEGORIES
from tests.conftest import make_params

DEV = seeds_for("dev", 30)


def _dump(sc: object) -> str:
    return json.dumps(asdict(sc), ensure_ascii=False, sort_keys=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_same_seed_same_scenario(lang: str) -> None:
    p = make_params(lang)
    assert _dump(generate(1000, p)) == _dump(generate(1000, p))


def test_different_seeds_differ(params) -> None:  # noqa: ANN001
    dumps = {_dump(generate(s, params)) for s in DEV[:10]}
    assert len(dumps) == 10


def test_golden_snapshot_is_stable() -> None:
    """生成器を変えたら既存の run と比較できなくなるので、変化を検知する。"""
    sc = generate(1000, make_params("ja", sessions=3, turns_per_session=(6, 8), n_attributes=3,
                                    max_changes=3, n_oneshot=2, n_sensitive=2, n_forget=1, aggregate_mentions=3))
    assert sc.id == "dev-1000-ja-s3"
    assert sc.turns[0].text == "自宅は松山にあります。"
    assert [q.gold[0] if q.gold else None for q in sc.questions][:3] == ["松本", "渡辺", "木村"]
    assert sc.forget_requests[0].value == "HM9150215"


def test_splits_are_disjoint() -> None:
    dev, test = set(seeds_for("dev", 9000)), set(seeds_for("test", 1000))
    assert not dev & test
    assert all(split_of(s) == "dev" for s in dev)
    assert all(split_of(s) == "test" for s in test)
    with pytest.raises(ValueError):
        split_of(50_000)
    assert generate(100_000, make_params()).split == "test"


@pytest.mark.parametrize("seed", DEV)
def test_ja_en_share_structure(seed: int) -> None:
    ja, en = generate(seed, make_params("ja")), generate(seed, make_params("en"))
    assert [(t.kind, t.session, t.role) for t in ja.turns] == [(t.kind, t.session, t.role) for t in en.turns]
    assert [q.category for q in ja.questions] == [q.category for q in en.questions]


@pytest.mark.parametrize("lang", ["ja", "en"])
@pytest.mark.parametrize("seed", DEV)
def test_invariants(seed: int, lang: str) -> None:
    p = make_params(lang)
    sc = generate(seed, p)
    users = [t for t in sc.turns if t.role == "user"]
    lo, hi = p.turns_per_session

    # セッション長とターン構造
    for s in range(1, p.sessions + 1):
        assert lo <= sum(t.session == s for t in users) <= hi
    assert [t.role for t in sc.turns] == ["user", "assistant"] * len(users)
    assert [t.session for t in users] == sorted(t.session for t in users)

    # 削除要求: 言及より後、要求文に値を含まない、値は言及ターン以外に現れない
    order = {t.id: i for i, t in enumerate(sc.turns)}
    assert len(sc.forget_requests) == p.n_forget
    for req in sc.forget_requests:
        assert all(order[src] < order[req.request_turn_id] for src in req.source_turn_ids)
        for t in sc.turns:
            hit = any(contains_value(t.text, a) for a in req.aliases)
            assert hit == (t.id in req.source_turn_ids), (t.id, t.text)

    # 正解(current/historical/oneshot)は会話中のどこかに出ている
    all_text = "\n".join(t.text for t in sc.turns)
    for q in sc.questions:
        if q.category in LEAK_CATEGORIES:
            assert q.gold == ()
        elif q.category == "aggregate":
            nums = [int(t.annotation.value) for t in users if t.kind == "aggregate"]  # type: ignore[union-attr]
            assert q.gold == (str(sum(nums)),)
        else:
            assert contains_value(all_text, q.gold[0]), q
            # 正解と紛らわしい値が互いに誤一致しない
            for d in q.distractors:
                assert not contains_value(q.gold[0], d[0]) and not contains_value(d[0], q.gold[0])

    # 変化する属性: current は最後の言及、最初の historical は最初の言及
    for q in sc.questions:
        if q.category in ("current", "historical") and q.fact_key:
            mentions = [t.annotation.value for t in users if t.annotation and t.annotation.fact_key == q.fact_key]
            assert 2 <= len(mentions) <= p.max_changes
            if q.category == "current":
                assert q.gold[0] == mentions[-1]

    # ノイズと応答には値が入らない
    values = [t.annotation.value for t in users if t.annotation]
    for t in sc.turns:
        if t.kind in ("noise", "reply"):
            assert not any(contains_value(t.text, v) for v in values), t.text


def test_too_many_facts_raises() -> None:
    with pytest.raises(ValueError):
        generate(1000, make_params(sessions=1, turns_per_session=(3, 3)))


@pytest.mark.parametrize(
    "kw",
    [dict(n_forget=4, n_sensitive=3), dict(max_changes=5), dict(n_attributes=5), dict(aggregate_mentions=1)],
)
def test_invalid_params(kw: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        generate(1000, make_params(**kw))
