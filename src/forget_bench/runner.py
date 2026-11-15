"""実験ランナー: 設定 → シナリオ生成 → 戦略ごとに observe/forget → 質問 → 採点 → runs/<run_id>/ に出力。

ランナーはバックエンドや戦略の種類で分岐しない(プラグインのレジストリ経由で作る)。
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import IO, Any

from forget_bench.backends import BackendError, MeteredBackend, make_backend
from forget_bench.backends.metered import RequestTags
from forget_bench.config import RunConfig, load_config
from forget_bench.memory import StrategyContext, make_strategy
from forget_bench.memory.base import estimate_memory_calls
from forget_bench.prompts import load_prompt
from forget_bench.report import render_summary_markdown
from forget_bench.scenario import generate
from forget_bench.scoring import score_answer
from forget_bench.scoring.summary import RequestRow, ScoreRow, summarize
from forget_bench.types import ForgetRequest, Scenario


def _json_default(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._f: IO[str] = path.open("a", encoding="utf-8")

    def write(self, obj: Any) -> None:
        self._f.write(_dumps(obj) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment() -> dict[str, Any]:
    pkgs = {}
    for name in ("forget-bench", "pydantic", "httpx", "pyyaml"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    status = _cmd(["git", "status", "--porcelain"])
    return {
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "git_dirty_files": status.splitlines() if status else [],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"]),
        "argv": sys.argv,
    }


def estimate_requests(cfg: RunConfig) -> dict[str, int]:
    """リクエスト数の見積もり(シナリオを生成して数える。LLMは呼ばない)。"""
    answer = memory = 0
    for lang in cfg.scenario.langs:
        for seed in cfg.scenario.resolved_seeds():
            sc = generate(seed, cfg.scenario.params(lang))
            n_user = sum(t.role == "user" for t in sc.turns)
            for s in cfg.strategies:
                answer += len(sc.questions)
                memory += estimate_memory_calls(s.name, n_user, s.params)
    return {"answer": answer, "memory": memory, "total": answer + memory}


def run_scenario_strategy(
    scenario: Scenario,
    strategy_name: str,
    strategy_label: str,
    strategy_params: dict[str, Any],
    backend: Any,
    cfg: RunConfig,
    request_sink: Any,
) -> tuple[list[ScoreRow], dict[str, Any]]:
    lang = scenario.params.lang
    metered = MeteredBackend(backend, request_sink, RequestTags(lang, scenario.id, strategy_label, "memory"))
    ctx = StrategyContext(
        lang=lang,
        system_prompt=load_prompt("system", lang),
        llm=metered,
        max_tokens=cfg.generation.max_tokens,
    )
    strategy = make_strategy(strategy_name, ctx, **strategy_params)

    t0 = time.perf_counter()
    forgotten: list[ForgetRequest] = []
    for turn in scenario.turns:
        strategy.observe(turn)
        req = scenario.forget_request_for_turn(turn.id)
        if req is not None:
            strategy.forget(req)
            forgotten.append(req)
    memory_s = time.perf_counter() - t0

    rows: list[ScoreRow] = []
    for q in scenario.questions:
        messages = strategy.build_context(q)
        answerer = metered.with_tags(purpose="answer", question_id=q.id)
        try:
            gen = answerer.generate(messages, max_tokens=cfg.generation.max_tokens, temperature=cfg.generation.temperature)
            answer, error = gen.text, None
        except BackendError as e:
            answer, error = None, str(e)
        s = score_answer(q, answer, tuple(forgotten), error=error)
        rows.append(
            ScoreRow(
                lang=lang, scenario_id=scenario.id, strategy=strategy_label, question_id=q.id,
                category=q.category, gold=list(q.gold), answer=s.answer, contains_gold=s.contains_gold,
                hit_distractor=s.hit_distractor, correct=s.correct, leaked=s.leaked,
                leaked_request_ids=list(s.leaked_request_ids), declared_deleted=s.declared_deleted, error=s.error,
            )
        )
    mem = {
        "lang": lang, "scenario_id": scenario.id, "strategy": strategy_label,
        "stats": asdict(strategy.stats()), "memory_build_s": memory_s,
        "total_s": time.perf_counter() - t0,
    }
    return rows, mem


def run(
    config_path: str | Path,
    *,
    out_root: str | Path = "runs",
    backend_override: str | None = None,
) -> Path:
    cfg, config_text = load_config(config_path)
    if backend_override is not None:
        cfg = cfg.model_copy(update={"backend": cfg.backend.model_copy(update={"name": backend_override, "params": {}})})
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.name}"
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenarios").mkdir()

    seeds = cfg.scenario.resolved_seeds()
    estimate = estimate_requests(cfg)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "config_path": str(config_path),
        "config_text": config_text,
        "config": cfg.model_dump(mode="json"),
        "seeds": seeds,
        "split": cfg.scenario.split,
        "request_estimate": estimate,
        **_environment(),
    }

    def write_meta() -> None:
        (out / "meta.json").write_text(_dumps(meta) + "\n", encoding="utf-8")

    write_meta()
    print(f"[forget-bench] run {run_id}: 見積もり {estimate['total']} リクエスト "
          f"(回答 {estimate['answer']} + メモリ構築 {estimate['memory']})", flush=True)

    requests_w = _JsonlWriter(out / "requests.jsonl")
    scores_w = _JsonlWriter(out / "scores.jsonl")
    progress_w = _JsonlWriter(out / "progress.jsonl")
    memory_w = _JsonlWriter(out / "memory.jsonl")
    request_rows: list[RequestRow] = []
    score_rows: list[ScoreRow] = []

    def sink(row: RequestRow) -> None:
        request_rows.append(row)
        requests_w.write(row)

    try:
        backend = make_backend(cfg.backend.name, **cfg.backend.params)
        warmup = getattr(backend, "warmup", None)
        if callable(warmup):
            warmup()
        meta["backend"] = backend.info()
        write_meta()

        forget_counts: dict[str, int] = {}
        total_units = len(cfg.scenario.langs) * len(seeds) * len(cfg.strategies)
        done = 0
        t_start = time.perf_counter()
        for lang in cfg.scenario.langs:
            for seed in seeds:
                scenario = generate(seed, cfg.scenario.params(lang))
                forget_counts[scenario.id] = len(scenario.forget_requests)
                (out / "scenarios" / f"{scenario.id}.json").write_text(_dumps(scenario), encoding="utf-8")
                for sc in cfg.strategies:
                    rows, mem = run_scenario_strategy(scenario, sc.name, sc.display, sc.params, backend, cfg, sink)
                    for r in rows:
                        scores_w.write(r)
                    score_rows.extend(rows)
                    memory_w.write(mem)
                    done += 1
                    progress = {
                        "at": datetime.now().astimezone().isoformat(),
                        "done": done, "total": total_units, "lang": lang, "scenario_id": scenario.id,
                        "strategy": sc.display, "n_correct": sum(r.correct for r in rows), "n_questions": len(rows),
                        "n_errors": sum(r.error is not None for r in rows), "elapsed_s": round(time.perf_counter() - t_start, 1),
                    }
                    progress_w.write(progress)
                    print(f"[forget-bench] {done}/{total_units} {scenario.id} {sc.display}: "
                          f"{progress['n_correct']}/{progress['n_questions']} correct, "
                          f"{progress['n_errors']} errors ({progress['elapsed_s']}s)", flush=True)

        summary = summarize(score_rows, request_rows, forget_counts)
        (out / "summary.json").write_text(_dumps({"run_id": run_id, "metrics": summary}) + "\n", encoding="utf-8")
        (out / "summary.md").write_text(render_summary_markdown(run_id, summary), encoding="utf-8")
        meta["status"] = "completed"
    except BaseException as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc()
        raise
    finally:
        meta["finished_at"] = datetime.now().astimezone().isoformat()
        meta["n_requests"] = len(request_rows)
        write_meta()
        for w in (requests_w, scores_w, progress_w, memory_w):
            w.close()
    return out
