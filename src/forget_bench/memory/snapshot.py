"""メモリの中身を JSON にできる形にする(デモ UI の表示用。ベンチマークの採点・集計では使わない)。

戦略の snapshot() が返す dict の形式("kind" で分かれる):
- "turns": 発話の列(full / full_redact / window)。items = [{id, session, role, text, redacted}]
- "text": 要約1本(summary)。text, pending_turns
- "facts": 事実ストア(facts_append)。items = [{seq, session, key, value, deleted, source}]
- "docs": 検索索引の発話(rag)。items = [{id, session, text}], removed
- "composite": 複数の部品(hybrid)。parts = [snapshot, ...]
- "kv": KV キャッシュのトークン範囲(kv_*)。spans = [{id, start, end, role, text, session}], drops = 切除の記録
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from forget_bench.types import Lang, Turn


def turns_snapshot(turns: Iterable[Turn], lang: Lang, *, redacted: set[str] | None = None,
                   note: str | None = None) -> dict[str, Any]:
    redacted = redacted or set()
    out: dict[str, Any] = {"kind": "turns", "items": [
        {"id": t.id, "session": t.session, "role": t.role, "text": t.text, "redacted": t.id in redacted} for t in turns]}
    if note:
        out["note"] = note
    return out
