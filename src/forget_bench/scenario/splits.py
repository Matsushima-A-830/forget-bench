"""dev / test 分割。seed の範囲で決める(調整用と最終評価用が混ざらないように)。

- dev:  1,000 〜 9,999
- test: 100,000 〜 999,999
プロンプトや戦略パラメータの調整には dev の seed だけを使う(AGENTS.md 絶対ルール4)。
"""

from __future__ import annotations

from forget_bench.types import Split

_RANGES: dict[str, tuple[int, int]] = {
    "dev": (1_000, 10_000),
    "test": (100_000, 1_000_000),
}


def split_of(seed: int) -> Split:
    for name, (lo, hi) in _RANGES.items():
        if lo <= seed < hi:
            return name  # type: ignore[return-value]
    raise ValueError(f"seed {seed} はどの分割にも属さない(dev: 1000-9999, test: 100000-999999)")


def seeds_for(split: Split, n: int, offset: int = 0) -> list[int]:
    """分割内の先頭から n 個の seed。"""
    lo, hi = _RANGES[split]
    if n < 0 or lo + offset + n > hi:
        raise ValueError("seed の範囲を超える")
    return list(range(lo + offset, lo + offset + n))
