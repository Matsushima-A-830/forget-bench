"""バックエンド呼び出しを1リクエスト1行で記録し、レスポンスをキャッシュするラッパー。

- メモリ戦略にはこのラッパーを purpose="memory" で渡すので、
  要約・抽出などメモリ構築のための呼び出しも自動的にコストに計上される(AGENTS.md)。
- キャッシュ: 鍵は (名前空間, messages, max_tokens, temperature, repeat) の SHA-256。
  同じ入力を二度送らない(CLAUDE.md「利用枠」)。repeat が違えば別の鍵なので、揺れの測定は妨げない。
  名前空間はバックエンドの設定(名前+パラメータ)から作るので、モデルや effort が違えば共有されない。
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from forget_bench.backends.base import Backend, BackendError
from forget_bench.scoring.summary import RequestRow
from forget_bench.types import Generation, Message


@dataclass(frozen=True)
class RequestTags:
    lang: str
    scenario_id: str
    strategy: str
    purpose: str
    question_id: str | None = None
    sessions: int = 0
    repeat: int = 0


class ResponseCache:
    """1レスポンス1ファイルのキャッシュ。書き込みは write_dir、読み込みは write_dir と read_dirs。"""

    def __init__(self, namespace: str, write_dir: Path | None, read_dirs: tuple[Path, ...] = ()) -> None:
        self.namespace = namespace
        self.write_dir = write_dir
        self.read_dirs = tuple(d for d in ((write_dir,) if write_dir else ()) + read_dirs)
        if write_dir is not None:
            write_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def key(self, messages: list[Message], max_tokens: int, temperature: float, repeat: int) -> str:
        payload = json.dumps(
            {
                "ns": self.namespace,
                "messages": [[m.role, m.content] for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repeat": repeat,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Generation | None:
        for d in self.read_dirs:
            path = d / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self.hits += 1
                return Generation(**data["generation"])
        with self._lock:
            self.misses += 1
        return None

    def put(self, key: str, gen: Generation) -> None:
        if self.write_dir is None:
            return
        path = self.write_dir / f"{key}.json"
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"namespace": self.namespace, "generation": asdict(gen)}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def cache_namespace(backend_name: str, params: dict[str, Any]) -> str:
    return json.dumps({"backend": backend_name, "params": params}, ensure_ascii=False, sort_keys=True)


class MeteredBackend:
    def __init__(
        self,
        inner: Backend,
        sink: Callable[[RequestRow], None],
        tags: RequestTags,
        cache: ResponseCache | None = None,
    ) -> None:
        self.inner = inner
        self.sink = sink
        self.tags = tags
        self.cache = cache
        self.name = inner.name

    def with_tags(self, **changes: Any) -> "MeteredBackend":
        return MeteredBackend(self.inner, self.sink, replace(self.tags, **changes), self.cache)

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float = 0.0) -> Generation:
        row: dict[str, Any] = {
            "lang": self.tags.lang,
            "sessions": self.tags.sessions,
            "repeat": self.tags.repeat,
            "scenario_id": self.tags.scenario_id,
            "strategy": self.tags.strategy,
            "purpose": self.tags.purpose,
            "question_id": self.tags.question_id,
            "n_messages": len(messages),
            "input_chars": sum(len(m.content) for m in messages),
        }
        key = None
        if self.cache is not None:
            key = self.cache.key(messages, max_tokens, temperature, self.tags.repeat)
            hit = self.cache.get(key)
            if hit is not None:
                self.sink(RequestRow(**row, input_tokens=hit.input_tokens, output_tokens=hit.output_tokens,
                                     latency_s=hit.latency_s, ttft_s=hit.ttft_s, output=hit.text,
                                     error=None, warnings=list(hit.warnings), cached=True))
                return hit
        try:
            gen = self.inner.generate(messages, max_tokens=max_tokens, temperature=temperature)
        except BackendError as e:
            self.sink(RequestRow(**row, input_tokens=None, output_tokens=None, latency_s=None,
                                 ttft_s=None, output=None, error=str(e), warnings=[]))
            raise
        if self.cache is not None and key is not None:
            self.cache.put(key, gen)
        self.sink(RequestRow(**row, input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
                             latency_s=gen.latency_s, ttft_s=gen.ttft_s, output=gen.text,
                             error=None, warnings=list(gen.warnings)))
        return gen

    def info(self) -> dict[str, Any]:
        return self.inner.info()
