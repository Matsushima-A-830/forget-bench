from __future__ import annotations

import json
from pathlib import Path

import yaml

from forget_bench.runner import run

CONFIG = {
    "name": "t",
    "backend": {"name": "dummy", "params": {"mode": "echo"}},
    "scenario": {"langs": ["ja", "en"], "split": "dev", "n_scenarios": 2, "sessions": 2,
                 "turns_per_session": [6, 8], "n_attributes": 2, "max_changes": 2, "n_oneshot": 1,
                 "n_sensitive": 2, "n_forget": 1, "aggregate_mentions": 2},
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
