"""揺れの幅: 図の各点の信頼区間と、戦略どうしの差の信頼区間を runs/ の生ログから出す(LLM を呼ばない)。

使い方: uv run python scripts/uncertainty.py [configs/figures.yaml] [--json out.json] [--boot 10000]

- 各点(バックエンド × 戦略、規模はまとめる)の 95% 区間
  - 覚える力: 質問単位(正解/不正解の比率)の Clopper–Pearson 区間
  - 忘れる力: 削除要求単位(漏れた/漏れない)の Clopper–Pearson 区間。0 件・全件でも幅が出る
  どちらも「質問・要求が互いに独立」とみなした区間なので、同じシナリオの質問どうしの相関は入っていない
- 戦略の差(同じバックエンドの中で、同じシナリオ・同じ質問について A − B)
  - 質問単位の対応ブートストラップ(質問を独立とみなす)
  - シナリオ単位のクラスタ・ブートストラップ(シナリオをまとめて引き直す。こちらのほうが保守的)
  区間が 0 をまたげば「この量では差があるとは言えない」と読む
- 乱数の seed は固定(--seed、既定 0)。同じ入力なら同じ区間が出る
- 数える単位・指標の定義は forget_bench.figures と同じ(完了した単位だけ)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from forget_bench.figures import (
    REMEMBER_CATEGORIES,
    _completed_units,
    _forget_counts,
    _jsonl,
    load_figures_config,
)

# --- 二項分布の正確な区間(scipy なし) -------------------------------------------


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """k/n の両側 (1−alpha) Clopper–Pearson 区間。"""
    if n == 0:
        return (math.nan, math.nan)

    def solve(f: Any) -> float:  # f は p について単調。二分法
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return (lower, upper)


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


# --- 集計 ----------------------------------------------------------------------


def load_rows(run: Path) -> list[dict[str, Any]]:
    done = _completed_units(run)
    return [s for s in _jsonl(run / "scores.jsonl") if (s["scenario_id"], s["strategy"], s.get("repeat", 0)) in done]


def point_intervals(run: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forget_n = _forget_counts(run)
    by_st: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_st[r["strategy"]].append(r)
    out = []
    for st, rs in sorted(by_st.items()):
        rem = [r for r in rs if r["category"] in REMEMBER_CATEGORIES]
        k_rem = sum(r["correct"] for r in rem)
        units = {(r["scenario_id"], r.get("repeat", 0)) for r in rs}
        n_req = sum(forget_n.get(sid, 0) for sid, _ in units)
        leaked = len({(r["scenario_id"], r.get("repeat", 0), rid) for r in rs for rid in r["leaked_request_ids"]})
        k_forget = n_req - leaked
        out.append({
            "strategy": st, "n_scenarios": len(units),
            "remember": {"k": k_rem, "n": len(rem), "p": k_rem / len(rem) if rem else None,
                         "ci95": clopper_pearson(k_rem, len(rem))},
            "forget": {"k": k_forget, "n": n_req, "p": k_forget / n_req if n_req else None,
                       "ci95": clopper_pearson(k_forget, n_req)},
        })
    return out


def paired_diffs(rows: list[dict[str, Any]], strategies: list[str], n_boot: int, seed: int) -> list[dict[str, Any]]:
    """覚える力の差 A − B(同じシナリオ・同じ質問で対応を取る)。"""
    table: dict[tuple[str, int, str], dict[str, int]] = defaultdict(dict)
    for r in rows:
        if r["category"] in REMEMBER_CATEGORIES:
            table[(r["scenario_id"], r.get("repeat", 0), r["question_id"])][r["strategy"]] = int(r["correct"])
    out = []
    for a, b in combinations(strategies, 2):
        pairs = [(k, v[a] - v[b]) for k, v in table.items() if a in v and b in v]
        if not pairs:
            continue
        diffs = [d for _, d in pairs]
        n = len(diffs)
        mean = sum(diffs) / n
        rng = random.Random(f"{seed}-{a}-{b}")
        q_boot = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
        clusters: dict[tuple[str, int], list[int]] = defaultdict(list)
        for (sid, rep, _), d in pairs:
            clusters[(sid, rep)].append(d)
        cl = list(clusters.values())
        c_boot = []
        for _ in range(n_boot):
            pick = [cl[rng.randrange(len(cl))] for _ in cl]
            tot = sum(len(c) for c in pick)
            c_boot.append(sum(sum(c) for c in pick) / tot)
        out.append({
            "a": a, "b": b, "n_questions": n, "n_scenarios": len(cl), "diff": mean,
            "only_a": sum(d > 0 for d in diffs), "only_b": sum(d < 0 for d in diffs),
            "ci95_question": (_percentile(q_boot, 0.025), _percentile(q_boot, 0.975)),
            "ci95_scenario": (_percentile(c_boot, 0.025), _percentile(c_boot, 0.975)),
        })
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/figures.yaml")
    ap.add_argument("--json")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_figures_config(args.config)
    result = []
    for ref in cfg.runs:
        run = Path(ref.run)
        rows = load_rows(run)
        points = point_intervals(run, rows)
        strategies = [p["strategy"] for p in points]
        diffs = paired_diffs(rows, strategies, args.boot, args.seed)
        result.append({"label": ref.label, "run": ref.run, "points": points, "diffs": diffs})
        print(f"\n## {ref.label}({run.name})\n")
        print("| strategy | scenarios | remember % [95% CI] (k/n) | forget % [95% CI] (k/n) |")
        print("|---|---:|---|---|")
        for p in points:
            r, f = p["remember"], p["forget"]
            print(f"| {p['strategy']} | {p['n_scenarios']} | {_pct(r['p'])} [{_pct(r['ci95'][0])}–{_pct(r['ci95'][1])}] "
                  f"({r['k']}/{r['n']}) | {_pct(f['p'])} [{_pct(f['ci95'][0])}–{_pct(f['ci95'][1])}] ({f['k']}/{f['n']}) |")
        print("\n| A | B | diff (pt) | 95% CI, question bootstrap | 95% CI, scenario bootstrap | only A / only B |")
        print("|---|---|---:|---|---|---|")
        for d in diffs:
            print(f"| {d['a']} | {d['b']} | {100 * d['diff']:+.1f} | [{100 * d['ci95_question'][0]:+.1f}, "
                  f"{100 * d['ci95_question'][1]:+.1f}] | [{100 * d['ci95_scenario'][0]:+.1f}, "
                  f"{100 * d['ci95_scenario'][1]:+.1f}] | {d['only_a']} / {d['only_b']} |")
    if args.json:
        Path(args.json).write_text(json.dumps({"boot": args.boot, "seed": args.seed, "runs": result},
                                              ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
