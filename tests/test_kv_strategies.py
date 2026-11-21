from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import yaml  # noqa: E402

from forget_bench.memory import StrategyContext, make_strategy  # noqa: E402
from forget_bench.scenario import generate  # noqa: E402
from forget_bench.scoring import contains_value  # noqa: E402
from tests.conftest import make_params  # noqa: E402
from tests.kv_fixtures import tiny_backend  # noqa: E402

SMALL = dict(sessions=3, turns_per_session=(3, 4), n_attributes=2, max_changes=2, n_oneshot=1,
             n_sensitive=2, n_forget=1, aggregate_mentions=2)


@pytest.fixture(scope="module")
def backend():  # noqa: ANN201
    return tiny_backend(prefill_chunk=64)


def _ctx(backend, lang: str = "ja") -> StrategyContext:  # noqa: ANN001
    return StrategyContext(lang, "SYSTEM", backend, max_tokens=4, kv=backend)


def _feed(s, sc) -> None:  # noqa: ANN001
    for t in sc.turns:
        s.observe(t)
        r = sc.forget_request_for_turn(t.id)
        if r:
            s.forget(r)


def _leaks(text: str, sc) -> bool:  # noqa: ANN001
    return any(contains_value(text, a) for r in sc.forget_requests for a in r.aliases)


def test_kv_full_keeps_everything(backend) -> None:  # noqa: ANN001
    sc = generate(1000, make_params(**SMALL))
    s = make_strategy("kv_full", _ctx(backend))
    _feed(s, sc)
    assert [sp.id for sp in s.handle.spans] == ["m0", *(t.id for t in sc.turns)]
    assert _leaks("\n".join(m.content for m in s.build_context(sc.questions[0])), sc)
    st = s.stats()
    assert st.kv_tokens == s.handle.length == st.prefill_tokens and st.dropped_tokens == 0


@pytest.mark.parametrize("name", ["kv_drop", "kv_drop_repos"])
def test_kv_drop_removes_forgotten_turns(backend, name: str) -> None:  # noqa: ANN001
    sc = generate(1000, make_params(**SMALL))
    s = make_strategy(name, _ctx(backend))
    _feed(s, sc)
    assert not _leaks("\n".join(m.content for m in s.build_context(sc.questions[0])), sc)
    removed = {sid for r in sc.forget_requests for sid in r.source_turn_ids}
    assert not removed & {sp.id for sp in s.handle.spans}
    st = s.stats()
    assert st.deleted_items == len(removed) and st.dropped_tokens > 0
    assert st.kv_tokens == st.prefill_tokens - st.dropped_tokens == s.handle.cache.get_seq_length()
    g = s.answer(sc.questions[0])
    assert s.handle.length == st.kv_tokens and g.output_tokens is not None


def test_drop_and_drop_repos_differ_only_between_gap_and_drop_time(backend) -> None:  # noqa: ANN001
    """違いが出るのは「切除した範囲より後ろにあり、切除の時点ですでにキャッシュにあった」発話だけ。

    切除より前の発話は同じ。切除の後に追記した発話は、どちらも詰めた位置で prefill するので層0のキーは同じ。
    """
    sc = generate(1000, make_params(**SMALL))
    a, b = make_strategy("kv_drop", _ctx(backend)), make_strategy("kv_drop_repos", _ctx(backend))
    _feed(a, sc)
    _feed(b, sc)
    req = sc.forget_requests[0]
    order = {t.id: i for i, t in enumerate(sc.turns)}
    gap = min(order[x] for x in req.source_turn_ids)
    drop_time = order[req.request_turn_id]  # この発話を observe した直後に forget される
    ka, kb = a.handle.cache.layers[0].keys, b.handle.cache.layers[0].keys
    for x, y in zip(a.handle.spans, b.handle.spans):
        same = torch.equal(ka[:, :, x.start:x.end], kb[:, :, y.start:y.end])
        expected_diff = x.id in order and gap < order[x.id] <= drop_time
        assert same is not expected_diff, x.id


def test_kv_evict_old_keeps_recent_sessions(backend) -> None:  # noqa: ANN001
    sc = generate(1001, make_params(**dict(SMALL, sessions=4)))
    s = make_strategy("kv_evict_old", _ctx(backend), max_sessions=2)
    _feed(s, sc)
    assert set(s.sessions.values()) == {3, 4}
    kept = {sp.id for sp in s.handle.spans}
    assert kept == {"m0"} | {t.id for t in sc.turns if t.session >= 3}
    assert s.stats().deleted_items == sum(t.session <= 2 for t in sc.turns)


def test_kv_strategy_requires_kv_backend() -> None:
    from forget_bench.backends import make_backend

    with pytest.raises(ValueError):
        make_strategy("kv_full", StrategyContext("ja", "S", make_backend("dummy"), 4))


def test_runner_with_kv_and_text_strategies_on_hf(tmp_path: Path, backend, monkeypatch) -> None:  # noqa: ANN001
    from forget_bench.backends import base
    from forget_bench.runner import run

    monkeypatch.setitem(base._REGISTRY, "hf-tiny", lambda **_: backend)
    cfg = {
        "name": "kv", "backend": {"name": "hf-tiny"}, "generation": {"max_tokens": 4},
        "retrieval": {"embedder": {"name": "hash"}},
        "scenario": dict({"langs": ["ja"], "split": "dev", "n_scenarios": 1}, **{k: list(v) if isinstance(v, tuple) else v for k, v in SMALL.items()}),
        "strategies": [{"name": "full"}, {"name": "kv_full"}, {"name": "kv_drop_repos"}, {"name": "kv_evict_old"}],
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    out = run(path, out_root=tmp_path / "runs")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["status"] == "completed" and meta["request_estimate"]["total"] == meta["n_requests"]
    reqs = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    kv_rows = [r for r in reqs if r["strategy"] == "kv_full"]
    assert kv_rows and all(r["context_tokens"] > r["input_tokens"] for r in kv_rows)  # キャッシュを再利用している
    mem = {json.loads(line)["strategy"]: json.loads(line)["stats"] for line in (out / "memory.jsonl").read_text().splitlines()}
    assert mem["kv_drop_repos"]["dropped_tokens"] > 0 and mem["kv_full"]["dropped_tokens"] == 0
    metrics = {m["strategy"]: m for m in json.loads((out / "summary.json").read_text())["metrics"]}
    assert metrics["kv_full"]["leak_rate_requests"] is not None
