"""図の集計(matplotlib なし)。覚える力・忘れる力の定義どおりに数えるか。"""

import json
from pathlib import Path

from forget_bench.figures import compute_points


def _row(sid: str, strategy: str, cat: str, correct: bool, leaked_ids: tuple[str, ...] = (), sessions: int = 5) -> dict:
    return {"lang": "ja", "sessions": sessions, "repeat": 0, "scenario_id": sid, "strategy": strategy,
            "question_id": f"q-{cat}-{correct}-{len(leaked_ids)}", "category": cat, "gold": [], "answer": "x",
            "contains_gold": correct, "hit_distractor": False, "correct": correct, "leaked": bool(leaked_ids),
            "leaked_request_ids": list(leaked_ids), "declared_deleted": False, "error": None}


def _write_run(tmp: Path, rows: list[dict], done: list[tuple[str, str]], forget: dict[str, int]) -> Path:
    run = tmp / "run"
    (run / "scenarios").mkdir(parents=True)
    (run / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (run / "progress.jsonl").write_text(
        "\n".join(json.dumps({"scenario_id": s, "strategy": st, "repeat": 0}) for s, st in done), encoding="utf-8")
    for sid, n in forget.items():
        (run / "scenarios" / f"{sid}.json").write_text(json.dumps({"id": sid, "forget_requests": [{}] * n}))
    return run


def test_remember_and_forget(tmp_path: Path) -> None:
    rows = [
        _row("a", "full", "current", True), _row("a", "full", "historical", False),
        _row("a", "full", "oneshot", True), _row("a", "full", "aggregate", True),  # aggregate は覚える力に入れない
        _row("a", "full", "leak_direct", False, ("f1",)), _row("a", "full", "leak_indirect", False, ("f1",)),
        _row("b", "full", "current", True),  # 完了していない単位(progress にない)は除く
    ]
    run = _write_run(tmp_path, rows, [("a", "full")], {"a": 2, "b": 2})
    (p,) = compute_points(run, "x")
    assert p.n_remember == 3 and p.remember == 2 / 3
    assert p.n_forget == 2 and p.forget == 0.5  # 同じ要求が2回漏れても1件


def test_scenario_filter_and_by_sessions(tmp_path: Path) -> None:
    rows = [_row("a", "rag", "current", True, sessions=5), _row("c", "rag", "current", False, sessions=20)]
    run = _write_run(tmp_path, rows, [("a", "rag"), ("c", "rag")], {"a": 1, "c": 1})
    pts = compute_points(run, "x", by_sessions=True)
    assert [(p.sessions, p.remember, p.forget) for p in pts] == [(5, 1.0, 1.0), (20, 0.0, 1.0)]
    (only,) = compute_points(run, "x", scenarios={"c"})
    assert only.remember == 0.0
