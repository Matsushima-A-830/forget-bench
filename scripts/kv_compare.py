"""KV 層とテキスト層の比較表を runs/<run_id>/ の生ログから作る(results/2026-11-21-week3-kv.md の表)。

使い方: uv run python scripts/kv_compare.py runs/<run_id>

- コスト表: 回答1回あたりの文脈トークン数(KV 層はキャッシュ長+質問)、回答の平均レイテンシ・TTFT、
  回答時のピーク VRAM、1シナリオあたりの所要時間、KV の prefill トークン数・切除トークン数
- 対応比較: 同じシナリオ・同じ質問で2つの戦略の回答を比べる(回答文の一致、正誤の一致、片方だけ正解)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAIRS = [("kv_full", "full"), ("kv_drop_repos", "kv_drop"), ("kv_drop_repos", "full_redact"), ("kv_drop", "kv_full")]


def load(run: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (run / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def cost_table(run: Path) -> str:
    answers: dict[tuple, list[dict]] = defaultdict(list)
    for r in load(run, "requests.jsonl"):
        if r["purpose"] == "answer" and r.get("error") is None:
            answers[(r["lang"], r["sessions"], r["strategy"])].append(r)
    units: dict[tuple, list[dict]] = defaultdict(list)
    for m in load(run, "memory.jsonl"):
        units[(m["lang"], m["sessions"], m["strategy"])].append(m)
    lines = ["| lang | sessions | strategy | ctx-tok/answer | latency s | TTFT s | peak VRAM MB | s/scenario | KV prefill tok | dropped tok |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(answers):
        rs = [r for r in answers[key] if not r.get("cached")]
        ctx = [r.get("context_tokens") or r["input_tokens"] for r in answers[key]]
        peak = [r["peak_vram_mb"] for r in answers[key] if r.get("peak_vram_mb")]
        us = units[key]
        kv = key[2].startswith("kv_")
        lines.append("| {} | {} | {} | {:,.0f} | {} | {} | {} | {:.1f} | {} | {} |".format(
            *key, mean(ctx),
            f"{mean(r['latency_s'] for r in rs):.2f}" if rs else "-",
            f"{mean(r['ttft_s'] for r in rs if r['ttft_s'] is not None):.2f}" if rs else "-",
            f"{max(peak):,.0f}" if peak else "-",
            mean(u["total_s"] for u in us),
            f"{mean(u['stats']['prefill_tokens'] for u in us):,.0f}" if kv else "-",
            f"{mean(u['stats']['dropped_tokens'] for u in us):,.0f}" if kv else "-",
        ))
    return "\n".join(lines)


def pair_table(run: Path) -> str:
    scores = {(s["scenario_id"], s["strategy"], s["question_id"]): s for s in load(run, "scores.jsonl")}
    lines = ["| A | B | lang | sessions | questions | same answer text | same correctness | only A correct | only B correct |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for a, b in PAIRS:
        acc: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        for (sid, st, q), s in scores.items():
            t = scores.get((sid, b, q))
            if st != a or t is None:
                continue
            c = acc[(s["lang"], s["sessions"])]
            c[0] += 1
            c[1] += s["answer"].strip() == t["answer"].strip()
            c[2] += s["correct"] == t["correct"]
            c[3] += s["correct"] and not t["correct"]
            c[4] += t["correct"] and not s["correct"]
        for (lang, n), c in sorted(acc.items()):
            lines.append(f"| {a} | {b} | {lang} | {n} | " + " | ".join(str(x) for x in c) + " |")
        tot = [sum(c[i] for c in acc.values()) for i in range(5)]
        lines.append(f"| {a} | {b} | all | all | " + " | ".join(f"**{x}**" for x in tot) + " |")
    return "\n".join(lines)


def main() -> None:
    run = Path(sys.argv[1])
    print("## コスト\n\n" + cost_table(run) + "\n\n## 対応比較(同じ質問)\n\n" + pair_table(run))


if __name__ == "__main__":
    main()
