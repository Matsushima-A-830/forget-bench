"""summary.json から Markdown の表を作る。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (見出し, キー, 書式)
    ("strategy", "strategy", "s"),
    ("current", "acc_current", "pct"),
    ("historical", "acc_historical", "pct"),
    ("oneshot", "acc_oneshot", "pct"),
    ("aggregate", "acc_aggregate", "pct"),
    ("leak/answer", "leak_rate_answers", "pct"),
    ("leak/request", "leak_rate_requests", "pct"),
    ("leak direct", "leak_rate_direct", "pct"),
    ("leak indirect", "leak_rate_indirect", "pct"),
    ("over-forget", "over_forget_rate", "pct"),
    ("in-tok answer", "answer_input_tokens", "int"),
    ("in-tok memory", "memory_input_tokens", "int"),
    ("mean latency s", "mean_answer_latency_s", "f2"),
    ("errors", "n_errors", "int"),
)


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "pct":
        return f"{100 * float(v):.1f}%"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "f2":
        return f"{float(v):.2f}"
    return str(v)


def _cell(m: dict[str, Any], key: str, kind: str) -> str:
    text = _fmt(m.get(key), kind)
    rng = m.get(f"range_{key}")
    if kind == "pct" and rng:
        text += f" ({100 * rng[0]:.0f}–{100 * rng[1]:.0f})"
    return text


def render_summary_markdown(run_id: str, metrics: list[dict[str, Any]]) -> str:
    lines = [f"# {run_id}", ""]
    if any(m.get("n_repeats", 1) > 1 for m in metrics):
        lines += ["繰り返しがある指標は「全繰り返しの平均 (繰り返しごとの最小–最大)」。", ""]
    groups = sorted({(str(m["lang"]), int(m.get("sessions", 0))) for m in metrics})
    for lang, sessions in groups:
        rows = [m for m in metrics if m["lang"] == lang and int(m.get("sessions", 0)) == sessions]
        n_q = {m["strategy"]: m["n_questions"] for m in rows}
        lines += [f"## lang={lang}, sessions={sessions}", "", f"質問数(戦略ごと、全繰り返し): {n_q}", ""]
        lines.append("| " + " | ".join(h for h, _, _ in _COLUMNS) + " |")
        lines.append("|" + "|".join("---" for _ in _COLUMNS) + "|")
        for m in rows:
            lines.append("| " + " | ".join(_cell(m, k, f) for _, k, f in _COLUMNS) + " |")
        lines.append("")
    return "\n".join(lines)


def report(run_dir: str | Path) -> str:
    data = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    return render_summary_markdown(data["run_id"], data["metrics"])


_PIVOT_ORDER = ("full", "full_redact", "window", "summary", "facts_append", "rag", "hybrid", "facts_append_oracle")
_PIVOT_COLS: tuple[tuple[str, str], ...] = (
    ("current", "acc_current"), ("historical", "acc_historical"), ("oneshot", "acc_oneshot"),
    ("aggregate", "acc_aggregate"), ("leak/request", "leak_rate_requests"), ("leak/answer", "leak_rate_answers"),
    ("over-forget", "over_forget_rate"),
)


def _pct(m: dict[str, Any], key: str) -> str:
    v = m.get(key)
    if v is None:
        return "—"
    text = f"{100 * float(v):.0f}"
    rng = m.get(f"range_{key}")
    if rng:
        text += f" ({100 * rng[0]:.0f}–{100 * rng[1]:.0f})"
    return text


def render_by_scale(metrics: list[dict[str, Any]]) -> str:
    """言語ごとに「戦略 × 規模」の1枚の表にする(結果メモ用)。値は %、括弧は繰り返し間の範囲。"""
    lines: list[str] = []
    strategies = sorted({str(m["strategy"]) for m in metrics},
                        key=lambda s: (_PIVOT_ORDER.index(s) if s in _PIVOT_ORDER else len(_PIVOT_ORDER), s))
    for lang in sorted({str(m["lang"]) for m in metrics}):
        lines += [f"#### lang={lang}", ""]
        lines.append("| strategy | sessions | " + " | ".join(c for c, _ in _PIVOT_COLS)
                     + " | in-tok/answer | in-tok memory | errors |")
        lines.append("|---|---:|" + "---:|" * (len(_PIVOT_COLS) + 3))
        for st in strategies:
            rows = sorted((m for m in metrics if m["lang"] == lang and m["strategy"] == st),
                          key=lambda m: int(m.get("sessions", 0)))
            for m in rows:
                calls = int(m.get("answer_calls") or 0)
                mean_in = int(m.get("answer_input_tokens") or 0) / calls if calls else 0
                lines.append(f"| {st} | {m.get('sessions')} | " + " | ".join(_pct(m, k) for _, k in _PIVOT_COLS)
                             + f" | {mean_in:,.0f} | {int(m.get('memory_input_tokens') or 0):,} | {m.get('n_errors')} |")
        lines.append("")
    return "\n".join(lines)


def metrics_from_logs(run_dir: str | Path, scenarios: set[str] | None = None) -> list[dict[str, Any]]:
    """scores.jsonl / requests.jsonl から集計し直す(途中で止まった run や、シナリオを絞った集計用)。

    完了していない(シナリオ × 戦略 × 繰り返し)の単位は、progress.jsonl に載っていないので除く。
    """
    from forget_bench.scoring.summary import RequestRow, ScoreRow, row_from_dict, summarize

    run_dir = Path(run_dir)

    def jsonl(name: str) -> list[dict[str, Any]]:
        path = run_dir / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    done = {(p["scenario_id"], p["strategy"], p.get("repeat", 0)) for p in jsonl("progress.jsonl")}

    def keep(d: dict[str, Any]) -> bool:
        unit = (d["scenario_id"], d["strategy"], d.get("repeat", 0))
        return unit in done and (scenarios is None or d["scenario_id"] in scenarios)

    scores = [row_from_dict(ScoreRow, d) for d in jsonl("scores.jsonl") if keep(d)]
    requests = [row_from_dict(RequestRow, d) for d in jsonl("requests.jsonl") if keep(d)]
    forget_counts: dict[str, int] = {}
    for path in (run_dir / "scenarios").glob("*.json"):
        sc = json.loads(path.read_text(encoding="utf-8"))
        forget_counts[sc["id"]] = len(sc["forget_requests"])
    return summarize(scores, requests, forget_counts)
