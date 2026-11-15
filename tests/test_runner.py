from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forget_bench.runner import RequestBudgetExceeded, estimate_requests, run
from forget_bench.config import RunConfig

CONFIG = {
    "name": "t",
    "backend": {"name": "dummy", "params": {"mode": "echo"}},
    "scenario": {"langs": ["ja", "en"], "split": "dev", "n_scenarios": 2, "sessions": 2,
                 "turns_per_session": [6, 8], "n_attributes": 2, "max_changes": 2, "n_oneshot": 1,
                 "n_sensitive": 2, "n_forget": 1, "aggregate_mentions": 2},
    "retrieval": {"embedder": {"name": "hash"}},
    "strategies": [
        {"name": "full"},
        {"name": "window", "params": {"n_turns": 3}},
        {"name": "facts_append", "label": "facts_oracle", "params": {"extractor": "oracle"}},
    ],
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_end_to_end_with_echo_backend(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    out = run(cfg, out_root=tmp_path / "runs")

    for name in ("meta.json", "requests.jsonl", "scores.jsonl", "summary.json", "summary.md", "progress.jsonl", "memory.jsonl"):
        assert (out / name).exists(), name
    meta = json.loads((out / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["seeds"] == [1000, 1001]
    assert meta["config_text"] == cfg.read_text(encoding="utf-8")
    assert meta["request_estimate"]["total"] == meta["n_requests"]
    assert len(list((out / "scenarios").glob("*.json"))) == 4

    scores = _jsonl(out / "scores.jsonl")
    requests = _jsonl(out / "requests.jsonl")
    assert len(scores) == len(requests)  # 回答用のみ(oracle なのでメモリ構築呼び出しなし)
    assert len(_jsonl(out / "progress.jsonl")) == 2 * 2 * 3

    metrics = {(m["lang"], m["strategy"]): m for m in json.loads((out / "summary.json").read_text())["metrics"]}
    for lang in ("ja", "en"):
        # echo は文脈をそのまま返すので: full は必ず漏れる、oracle の事実ストアは漏れない
        assert metrics[(lang, "full")]["leak_rate_requests"] == 1.0
        assert metrics[(lang, "facts_oracle")]["leak_rate_answers"] == 0.0
        assert metrics[(lang, "facts_oracle")]["n_errors"] == 0


def test_backend_override(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    data = dict(CONFIG, backend={"name": "ollama", "params": {"model": "x", "num_ctx": 1}})
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    out = run(cfg, out_root=tmp_path / "runs", backend_override="dummy")
    assert json.loads((out / "meta.json").read_text())["backend"]["backend"] == "dummy"


def _write(tmp_path: Path, data: dict) -> Path:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return cfg


SWEEP = dict(CONFIG, scenario=dict(CONFIG["scenario"], langs=["ja"], n_scenarios=1, sessions=[2, 3]), repeats=2)


def test_sweep_and_repeats(tmp_path: Path) -> None:
    out = run(_write(tmp_path, SWEEP), out_root=tmp_path / "runs")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["sessions"] == [2, 3]
    assert meta["request_estimate"]["total"] == meta["n_requests"]
    metrics = json.loads((out / "summary.json").read_text())["metrics"]
    assert {(m["sessions"], m["strategy"]) for m in metrics} == {(s, k) for s in (2, 3) for k in ("full", "window", "facts_oracle")}
    assert all(m["n_repeats"] == 2 for m in metrics)
    assert "sessions=3" in (out / "summary.md").read_text()
    # 繰り返しは鍵が違うので、同じ run の中では1度もキャッシュに当たらない
    assert meta["n_cache_hits"] == 0


def test_reuse_cache_avoids_resending(tmp_path: Path) -> None:
    cfg = _write(tmp_path, SWEEP)
    first = run(cfg, out_root=tmp_path / "runs")
    second = run(cfg, out_root=tmp_path / "runs2", reuse_cache=[first])
    m1, m2 = (json.loads((d / "meta.json").read_text()) for d in (first, second))
    assert m2["n_cache_hits"] == m2["n_requests"] == m1["n_requests"]
    assert m2["n_backend_calls"] == 0
    assert (first / "scores.jsonl").read_text() == (second / "scores.jsonl").read_text()


def test_max_requests_stops_before_running(tmp_path: Path) -> None:
    data = dict(SWEEP, max_requests=10)
    with pytest.raises(RequestBudgetExceeded):
        run(_write(tmp_path, data), out_root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
    est = estimate_requests(RunConfig.model_validate(data))
    out = run(_write(tmp_path, data), out_root=tmp_path / "runs", max_requests=est["total"])
    assert out.exists()


def test_concurrency_gives_same_scores(tmp_path: Path) -> None:
    seq = run(_write(tmp_path, SWEEP), out_root=tmp_path / "seq")
    par = run(_write(tmp_path, dict(SWEEP, concurrency=4)), out_root=tmp_path / "par")

    def key(d: Path) -> list[str]:
        return sorted((d / "scores.jsonl").read_text().splitlines())

    assert key(seq) == key(par)
    assert json.loads((par / "meta.json").read_text())["n_requests"] == json.loads((seq / "meta.json").read_text())["n_requests"]
    def metrics(d: Path) -> list[dict]:
        # レイテンシ(実測の時間)以外は一致する
        return [{k: v for k, v in m.items() if "latency" not in k and "ttft" not in k}
                for m in json.loads((d / "summary.json").read_text())["metrics"]]

    assert metrics(seq) == metrics(par)


def test_fatal_error_stops_run(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from forget_bench.backends import BackendFatalError
    from forget_bench.backends.dummy import DummyBackend

    calls = {"n": 0}

    def boom(self, messages, *, max_tokens, temperature=0.0):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] > 5:
            raise BackendFatalError("quota")
        return DummyBackend.__dict__["_orig"](self, messages, max_tokens=max_tokens, temperature=temperature)

    monkeypatch.setattr(DummyBackend, "_orig", DummyBackend.generate, raising=False)
    monkeypatch.setattr(DummyBackend, "generate", boom)
    for conc in (1, 3):
        with pytest.raises(BackendFatalError):
            run(_write(tmp_path, dict(SWEEP, concurrency=conc)), out_root=tmp_path / f"f{conc}")
        (d,) = (tmp_path / f"f{conc}").iterdir()
        meta = json.loads((d / "meta.json").read_text())
        assert meta["status"] == "failed" and "quota" in meta["error"]
        calls["n"] = 0


def test_report_from_logs_matches_summary(tmp_path: Path) -> None:
    from forget_bench.report import metrics_from_logs, render_by_scale

    out = run(_write(tmp_path, SWEEP), out_root=tmp_path / "runs")
    summary = json.loads((out / "summary.json").read_text())["metrics"]
    assert metrics_from_logs(out) == summary
    table = render_by_scale(summary)
    assert "| full | 2 |" in table and "| full | 3 |" in table
    one = metrics_from_logs(out, {"dev-1000-ja-s2"})
    assert {m["sessions"] for m in one} == {2}
