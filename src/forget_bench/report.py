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


def render_summary_markdown(run_id: str, metrics: list[dict[str, Any]]) -> str:
    lines = [f"# {run_id}", ""]
    for lang in sorted({str(m["lang"]) for m in metrics}):
        rows = [m for m in metrics if m["lang"] == lang]
        n_q = {m["strategy"]: m["n_questions"] for m in rows}
        lines += [f"## lang={lang}", "", f"質問数(戦略ごと): {n_q}", ""]
        lines.append("| " + " | ".join(h for h, _, _ in _COLUMNS) + " |")
        lines.append("|" + "|".join("---" for _ in _COLUMNS) + "|")
        for m in rows:
            lines.append("| " + " | ".join(_fmt(m.get(k), f) for _, k, f in _COLUMNS) + " |")
        lines.append("")
    return "\n".join(lines)


def report(run_dir: str | Path) -> str:
    data = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    return render_summary_markdown(data["run_id"], data["metrics"])
