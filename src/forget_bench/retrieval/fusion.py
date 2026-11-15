"""順位融合(Reciprocal Rank Fusion)。"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """複数の順位リストを融合し、スコアの高い順(同点は最初に現れた順)に返す。"""
    score: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(score, key=lambda d: (-score[d], first_seen[d]))
