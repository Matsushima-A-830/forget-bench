"""発表用の図を runs/ の生ログから作る(docs/design.md「4. 発表で見せたい図」)。

- 図の元データは scores.jsonl(採点結果)と requests.jsonl(コスト)だけ。図に載る数値は figures.json にも書き出し、
  どの run・どの commit から来たかを一緒に残す(AGENTS.md 絶対ルール1・3)。
- 指標の定義:
  - 覚える力 = current / historical / oneshot の質問をまとめた正答率(aggregate はどのローカルモデルもほぼ 0 なので含めない)
  - 忘れる力 = 1 − 削除要求あたりの漏洩率(削除要求ごとに、その値が1回でも回答に出たら漏洩)
- 集計(points)は matplotlib なしで動く。描画だけ viz extra(matplotlib)が要る。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REMEMBER_CATEGORIES = ("current", "historical", "oneshot")


@dataclass(frozen=True)
class RunRef:
    label: str  # 図に出すバックエンド名(例: "ollama qwen2.5:7b")
    run: str  # runs/<run_id>


@dataclass
class FiguresConfig:
    runs: list[RunRef]
    common_scenarios: list[str] = field(default_factory=list)  # 全バックエンドに共通するシナリオ(公平な比較用)
    line_strategies: list[str] = field(default_factory=list)  # 規模曲線に描く戦略(色はこの順で固定)


def load_figures_config(path: str | Path) -> FiguresConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return FiguresConfig(
        runs=[RunRef(**r) for r in raw["runs"]],
        common_scenarios=list(raw.get("common_scenarios") or []),
        line_strategies=list(raw.get("line_strategies") or []),
    )


@dataclass(frozen=True)
class Point:
    backend: str
    strategy: str
    sessions: int | None  # None = 規模をまとめた値
    n_remember: int  # 覚える力の分母(質問数)
    remember: float | None
    n_forget: int  # 忘れる力の分母(削除要求数)
    forget: float | None
    over_forget: float | None  # 削除要求していない情報を「削除済み」と答えて外した率(参考)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _completed_units(run: Path) -> set[tuple[str, str, int]]:
    return {(p["scenario_id"], p["strategy"], p.get("repeat", 0)) for p in _jsonl(run / "progress.jsonl")}


def _forget_counts(run: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in (run / "scenarios").glob("*.json"):
        sc = json.loads(path.read_text(encoding="utf-8"))
        out[sc["id"]] = len(sc["forget_requests"])
    return out


def compute_points(
    run: str | Path, backend: str, scenarios: set[str] | None = None, by_sessions: bool = False
) -> list[Point]:
    """1つの run から (戦略[, 規模]) ごとの点を作る。完了していない単位(停止した run の途中)は除く。"""
    run = Path(run)
    done = _completed_units(run)
    forget_n = _forget_counts(run)
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for s in _jsonl(run / "scores.jsonl"):
        if (s["scenario_id"], s["strategy"], s.get("repeat", 0)) not in done:
            continue
        if scenarios is not None and s["scenario_id"] not in scenarios:
            continue
        groups[(s["strategy"], s["sessions"] if by_sessions else None)].append(s)
    points = []
    for (strategy, sessions), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        rem = [r for r in rows if r["category"] in REMEMBER_CATEGORIES]
        units = {(r["scenario_id"], r.get("repeat", 0)) for r in rows}
        n_req = sum(forget_n.get(sid, 0) for sid, _ in units)
        leaked = {(r["scenario_id"], r.get("repeat", 0), rid) for r in rows for rid in r["leaked_request_ids"]}
        normal = [r for r in rows if r["answer"] is not None and r["category"] in REMEMBER_CATEGORIES + ("aggregate",)]
        points.append(Point(
            backend=backend, strategy=strategy, sessions=sessions,
            n_remember=len(rem), remember=sum(r["correct"] for r in rem) / len(rem) if rem else None,
            n_forget=n_req, forget=1 - len(leaked) / n_req if n_req else None,
            over_forget=(sum(r["declared_deleted"] and not r["correct"] for r in normal) / len(normal)) if normal else None,
        ))
    return points


def answer_latency(run: str | Path, strategies: list[str]) -> dict[str, dict[int, float]]:
    """戦略 → 規模 → 回答1回の平均レイテンシ(秒)。キャッシュから再生したものは除く。"""
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in _jsonl(Path(run) / "requests.jsonl"):
        if r["purpose"] == "answer" and r["strategy"] in strategies and r["latency_s"] is not None and not r.get("cached"):
            acc[r["strategy"]][r["sessions"]].append(r["latency_s"])
    return {st: {n: sum(v) / len(v) for n, v in sorted(d.items())} for st, d in acc.items()}


def _run_meta(run: str | Path) -> dict[str, Any]:
    m = json.loads((Path(run) / "meta.json").read_text(encoding="utf-8"))
    backend = m.get("backend") or {}
    return {"run_id": m.get("run_id"), "status": m.get("status"), "git_commit": m.get("git_commit"),
            "git_dirty": m.get("git_dirty"), "model": backend.get("model"), "seeds": m.get("seeds"),
            "sessions": m.get("sessions")}


# --- 描画 -------------------------------------------------------------------------

# 既定パレット(dataviz スキルの検証済み8色、ライト面 #fcfcfb)。系列の色は戦略に固定し、順位では塗り替えない
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LAYER_COLOR = {"text": SERIES[0], "kv": SERIES[1]}


def _layer(strategy: str) -> str:
    return "kv" if strategy.startswith("kv_") else "text"


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    families = {f.name for f in font_manager.fontManager.ttflist}
    jp = next((f for f in ("IPAGothic", "IPAPGothic", "Noto Sans CJK JP", "Noto Sans JP") if f in families), None)
    plt.rcParams.update({
        "font.family": [jp, "DejaVu Sans"] if jp else ["DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 1, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "font.size": 10, "svg.fonttype": "none",
    })
    return plt


def _spread(ys: list[float], min_gap: float) -> list[float]:
    """ラベルの縦位置を、順序を保ったまま min_gap 以上離す(直接ラベルの重なりを避ける)。"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for prev, cur in zip(order, order[1:]):
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def _place_labels(ax: Any, items: list[tuple[str, float, float]], fontsize: float = 8) -> None:
    """点の直接ラベルを、既に置いたラベルと重ならない位置に置く(上下に段をずらし、離れたら引き出し線を付ける)。

    データは動かさない(点の位置はそのまま。ずらすのはラベルだけ)。
    """
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    pt = fig.dpi / 72  # 1pt あたりのピクセル
    char_w, h = fontsize * 0.62 * pt, fontsize * 1.25 * pt
    offsets = [10, -18, 26, -34, 42, -50, 58, -66, 74]
    # 点そのものも障害物にする(近い2点で、ラベルが隣の点に寄り添って取り違えて読まれるのを防ぐ)
    r = 6 * pt
    placed: list[tuple[float, float, float, float]] = [
        (px - r, py - r, px + r, py + r) for px, py in (to_px((x, y)) for _, x, y in items)
    ]
    for text, x, y in sorted(items, key=lambda t: (t[2], t[1])):
        px, py = to_px((x, y))
        w = len(text) * char_w
        for dy in offsets:
            cy = py + dy * pt
            box = (px - w / 2, cy - h / 2, px + w / 2, cy + h / 2)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                break
        placed.append(box)
        arrow = {"arrowstyle": "-", "color": TEXT_2, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 4} if abs(dy) > 18 else None
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, dy), ha="center", va="center",
                    fontsize=fontsize, color=TEXT_2, arrowprops=arrow)


def plot_tradeoff(points: list[Point], backends: list[str], title: str, path: Path) -> None:
    """覚える力(横)× 忘れる力(縦)。バックエンドごとに1枚(小さな複数図)、点は戦略。色は層(テキスト/KV)。"""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, len(backends), figsize=(4.6 * len(backends), 5.4), sharex=True, sharey=True,
                             squeeze=False)
    for ax, backend in zip(axes[0], backends):
        pts = [p for p in points if p.backend == backend and p.remember is not None and p.forget is not None
               and p.strategy != "facts_append_oracle"]
        n_rem = max((p.n_remember for p in pts), default=0)
        n_fg = max((p.n_forget for p in pts), default=0)
        ax.set_title(f"{backend}\n(質問 {n_rem} 問 / 削除要求 {n_fg} 件 ずつ)", loc="left", fontsize=10)
        for layer in ("text", "kv"):
            lp = [p for p in pts if _layer(p.strategy) == layer]
            if lp:
                # 層は色と形の両方で区別する(テキスト層 = 丸、KV 層 = ひし形。同じ値の点が重なっても両方見える)
                ax.scatter([100 * p.remember for p in lp], [100 * p.forget for p in lp],
                           s=64 if layer == "text" else 44, marker="o" if layer == "text" else "D",
                           zorder=3 if layer == "text" else 4,
                           color=LAYER_COLOR[layer], edgecolors=SURFACE, linewidths=1.5,
                           label="テキスト層" if layer == "text" else "KV層")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-8, 140)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        _place_labels(ax, [(p.strategy, 100 * p.remember, 100 * p.forget) for p in pts])
        ax.set_xlabel("覚える力: current+historical+oneshot 正答率 (%)")
    axes[0][0].set_ylabel("忘れる力: 100 − 削除要求あたりの漏洩率 (%)")
    handles, labels = [], []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def plot_scale(points: list[Point], backends: list[str], strategies: list[str], title: str, path: Path) -> None:
    """規模(セッション数)ごとの覚える力。バックエンドごとに1枚、線は戦略(色は strategies の順で固定)。"""
    plt = _setup_matplotlib()
    colors = {s: SERIES[i] for i, s in enumerate(strategies)}
    fig, axes = plt.subplots(1, len(backends), figsize=(4.6 * len(backends), 4.6), sharey=True, squeeze=False)
    for ax, backend in zip(axes[0], backends):
        ends: list[tuple[str, float, float]] = []
        n = 0
        for st in strategies:
            pts = sorted((p for p in points if p.backend == backend and p.strategy == st and p.remember is not None),
                         key=lambda p: p.sessions or 0)
            if not pts:
                continue
            n = max(n, max(p.n_remember for p in pts))
            xs, ys = [p.sessions for p in pts], [100 * p.remember for p in pts]
            ax.plot(xs, ys, color=colors[st], linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=st)
            ax.scatter(xs, ys, color=colors[st], s=40, edgecolors=SURFACE, linewidths=2, zorder=3)
            ends.append((st, xs[-1], ys[-1]))
        label_ys = _spread([y for _, _, y in ends], 6.0)
        for (st, x, _), ly in zip(ends, label_ys):
            ax.annotate(st, (x, ly), textcoords="offset points", xytext=(6, -3), fontsize=8, color=TEXT_2)
        ax.set_xscale("log")
        ax.set_xticks([5, 20, 50], ["5", "20", "50"])
        ax.minorticks_off()
        ax.set_xlim(4, 110)
        ax.set_ylim(-5, 105)
        ax.set_title(f"{backend}\n(各点 最大 {n} 問)", loc="left", fontsize=10)
        ax.set_xlabel("セッション数")
    axes[0][0].set_ylabel("覚える力 (%)")
    handles, labels = [], []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    _save(fig, path)


_LATENCY_LABEL = {"full": "full(質問ごとに文脈全体を prefill)", "kv_full": "kv_full(会話の KV を使い回す)"}


def plot_latency(lat: dict[str, dict[int, float]], strategies: list[str], title: str, path: Path) -> None:
    """回答1回のレイテンシ(秒、対数軸)を規模ごとに。テキスト層の full と KV 層の kv_full の比較用。"""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for st in strategies:
        d = lat.get(st)
        if not d:
            continue
        xs, ys = list(d), list(d.values())
        color = LAYER_COLOR[_layer(st)]
        ax.plot(xs, ys, color=color, linewidth=2, label=_LATENCY_LABEL.get(st, st))
        ax.scatter(xs, ys, color=color, s=40, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{st}  {ys[-1]:.1f} 秒", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, -3),
                    fontsize=9, color=TEXT_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 20, 50], ["5", "20", "50"])
    ax.minorticks_off()
    ax.set_xlim(4, 160)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("セッション数")
    ax.set_ylabel("回答1回の平均レイテンシ (秒、対数)")
    ax.legend(frameon=False, loc="upper left", fontsize=9, handlelength=2.5, borderaxespad=1.0)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    _save(fig, path)


def _save(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def make_figures(config_path: str | Path, out_dir: str | Path) -> Path:
    cfg = load_figures_config(config_path)
    out = Path(out_dir)
    backends = [r.label for r in cfg.runs]
    all_pts: list[Point] = []
    common_pts: list[Point] = []
    scale_pts: list[Point] = []
    for ref in cfg.runs:
        all_pts += compute_points(ref.run, ref.label)
        if cfg.common_scenarios:
            common_pts += compute_points(ref.run, ref.label, set(cfg.common_scenarios))
        scale_pts += compute_points(ref.run, ref.label, by_sessions=True)
    plot_tradeoff(all_pts, backends, "覚える力 × 忘れる力(各バックエンドの全シナリオ、規模をまとめた値)",
                  out / "tradeoff_all")
    if common_pts:
        plot_tradeoff(common_pts, backends, "覚える力 × 忘れる力(全バックエンド共通のシナリオだけ)", out / "tradeoff_common")
    plot_scale(scale_pts, backends, cfg.line_strategies, "規模に対する覚える力", out / "scale_remember")
    hf = next((r for r in cfg.runs if "hf" in r.label), None)
    lat: dict[str, dict[int, float]] = {}
    if hf is not None:
        lat = answer_latency(hf.run, ["full", "kv_full"])
        plot_latency(lat, ["full", "kv_full"], f"回答1回のレイテンシ({hf.label})",
                     out / "latency_kv")
    data = {
        "definitions": {
            "remember": "current / historical / oneshot の質問をまとめた正答率",
            "forget": "1 - 削除要求あたりの漏洩率",
            "over_forget": "削除要求していない情報を「削除済み」と答えて外した率(current/historical/oneshot/aggregate)",
        },
        "config": str(config_path),
        "runs": {r.label: _run_meta(r.run) for r in cfg.runs},
        "common_scenarios": cfg.common_scenarios,
        "points_all": [asdict(p) for p in all_pts],
        "points_common": [asdict(p) for p in common_pts],
        "points_by_sessions": [asdict(p) for p in scale_pts],
        "answer_latency_s": lat,
    }
    (out / "figures.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
