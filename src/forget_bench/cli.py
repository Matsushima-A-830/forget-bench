"""コマンドライン: forget-bench run / report / scenario。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from forget_bench.report import report
from forget_bench.config import load_config
from forget_bench.runner import estimate_requests, run
from forget_bench.scenario import generate, split_of
from forget_bench.types import ScenarioParams


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forget-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="設定ファイルで実験を回す")
    pr.add_argument("config")
    pr.add_argument("--out", default="runs")
    pr.add_argument("--backend", default=None, help="バックエンドを差し替える(例: dummy でドライラン)")
    pr.add_argument("--reuse-cache", action="append", default=None, metavar="RUN_DIR",
                    help="以前の run のキャッシュを再利用する(同じ入力を再送しない)。複数指定可")
    pr.add_argument("--max-requests", type=int, default=None,
                    help="見積もりリクエスト数の上限(設定の max_requests を上書き)")
    pc = sub.add_parser("probe-claude", help="claude バックエンドの隔離を確認する(4リクエスト消費)")
    pc.add_argument("--model", default="claude-opus-5-5")
    pc.add_argument("--effort", default="low")
    pc.add_argument("--auth", choices=["token", "login"], default="token")
    pc.add_argument("--out", default="runs")
    pe = sub.add_parser("estimate", help="LLMを呼ばずにリクエスト数を見積もる")
    pe.add_argument("config")

    pp = sub.add_parser("report", help="runs/<run_id> の表を表示する")
    pp.add_argument("run_dir")
    pp.add_argument("--by-scale", action="store_true", help="言語ごとに「戦略 × 規模」の1枚の表にする")
    pp.add_argument("--from-logs", action="store_true",
                    help="summary.json ではなく生ログから集計する(途中で止まった run 用。完了した単位だけ)")
    pp.add_argument("--scenarios", default=None, help="集計するシナリオIDをカンマ区切りで絞る(--from-logs と併用)")

    ps = sub.add_parser("scenario", help="シナリオを1つ生成してJSONで表示する(dev分割のみ)")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--lang", choices=["ja", "en"], default="ja")
    ps.add_argument("--sessions", type=int, default=5)
    ps.add_argument("--turns", type=int, nargs=2, default=(10, 20), metavar=("MIN", "MAX"))

    args = p.parse_args(argv)
    if args.cmd == "run":
        out = run(args.config, out_root=args.out, backend_override=args.backend,
                  reuse_cache=args.reuse_cache, max_requests=args.max_requests)
        print(report(out))
        print(f"出力: {out}")
    elif args.cmd == "probe-claude":
        from pathlib import Path

        from forget_bench.backends.claude import ClaudeCodeBackend
        from forget_bench.backends.claude_probe import run_probe

        checks, out = run_probe(ClaudeCodeBackend(args.model, effort=args.effort, auth=args.auth), Path(args.out))
        for c in checks:
            print(f"[{'ok' if c.passed else 'NG'}] {c.name}: {c.detail}")
        print(f"生データ: {out}/probe.json")
        return 0 if all(c.passed for c in checks) else 1
    elif args.cmd == "estimate":
        cfg, _ = load_config(args.config)
        print(json.dumps(estimate_requests(cfg), ensure_ascii=False))
    elif args.cmd == "report":
        from pathlib import Path

        from forget_bench.report import metrics_from_logs, render_by_scale, render_summary_markdown

        if args.from_logs or args.scenarios:
            wanted = set(args.scenarios.split(",")) if args.scenarios else None
            metrics = metrics_from_logs(args.run_dir, wanted)
        else:
            metrics = json.loads((Path(args.run_dir) / "summary.json").read_text(encoding="utf-8"))["metrics"]
        print(render_by_scale(metrics) if args.by_scale else render_summary_markdown(Path(args.run_dir).name, metrics))
    elif args.cmd == "scenario":
        if split_of(args.seed) != "dev":
            # test 分割の中身を見ながらプロンプトを調整しないため(AGENTS.md 絶対ルール4)
            print("test 分割のシナリオは表示しない。dev の seed (1000-9999) を使う。", file=sys.stderr)
            return 2
        params = ScenarioParams(
            lang=args.lang, sessions=args.sessions, turns_per_session=tuple(args.turns),
            n_attributes=4, max_changes=4, n_oneshot=3, n_sensitive=3, n_forget=2, aggregate_mentions=4,
        )
        print(json.dumps(asdict(generate(args.seed, params)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
