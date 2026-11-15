"""コマンドライン: forget-bench run / report / scenario。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from forget_bench.report import report
from forget_bench.runner import run
from forget_bench.scenario import generate, split_of
from forget_bench.types import ScenarioParams


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forget-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="設定ファイルで実験を回す")
    pr.add_argument("config")
    pr.add_argument("--out", default="runs")
    pr.add_argument("--backend", default=None, help="バックエンドを差し替える(例: dummy でドライラン)")

    pp = sub.add_parser("report", help="runs/<run_id> の表を表示する")
    pp.add_argument("run_dir")

    ps = sub.add_parser("scenario", help="シナリオを1つ生成してJSONで表示する(dev分割のみ)")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--lang", choices=["ja", "en"], default="ja")
    ps.add_argument("--sessions", type=int, default=5)
    ps.add_argument("--turns", type=int, nargs=2, default=(10, 20), metavar=("MIN", "MAX"))

    args = p.parse_args(argv)
    if args.cmd == "run":
        out = run(args.config, out_root=args.out, backend_override=args.backend)
        print(report(out))
        print(f"出力: {out}")
    elif args.cmd == "report":
        print(report(args.run_dir))
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
