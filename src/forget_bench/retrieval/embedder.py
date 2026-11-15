"""埋め込みモデル。テキストごとにメモ化し、同じ文を二度埋め込まない(rag と hybrid で共有)。"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import httpx

from forget_bench.retrieval.bm25 import tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def info(self) -> dict[str, Any]: ...

    def stats(self) -> dict[str, int]: ...


class EmbedderError(RuntimeError):
    pass


_REGISTRY: dict[str, Callable[..., Embedder]] = {}


def register_embedder(name: str) -> Callable[[Callable[..., Embedder]], Callable[..., Embedder]]:
    def deco(factory: Callable[..., Embedder]) -> Callable[..., Embedder]:
        _REGISTRY[name] = factory
        return factory

    return deco


def make_embedder(name: str, **params: Any) -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(f"未知の埋め込み: {name}(登録済み: {sorted(_REGISTRY)})")
    return _REGISTRY[name](**params)


def available_embedders() -> list[str]:
    return sorted(_REGISTRY)


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class _Memo:
    def __init__(self) -> None:
        self.memo: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.texts = 0
        self.memo_hits = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:  # 並列実行時も同じ文を二度埋め込まない
            missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
            self.memo_hits += len(texts) - len(missing)
            if missing:
                self.calls += 1
                self.texts += len(missing)
                for t, v in zip(missing, self._embed(missing)):
                    self.memo[t] = _normalize(v)
            return [self.memo[t] for t in texts]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def stats(self) -> dict[str, int]:
        return {"calls": self.calls, "texts": self.texts, "memo_hits": self.memo_hits}


@register_embedder("ollama")
class OllamaEmbedder(_Memo):
    name = "ollama"

    def __init__(self, model: str = "bge-m3", host: str = "http://localhost:11434", batch_size: int = 64,
                 timeout_s: float = 300.0, client: httpx.Client | None = None, **_: Any) -> None:
        super().__init__()
        self.model = model
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout_s)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                resp = self._client.post(f"{self.host}/api/embed", json={"model": self.model, "input": batch})
                resp.raise_for_status()
                vecs = resp.json()["embeddings"]
            except (httpx.HTTPError, ValueError, KeyError) as e:
                raise EmbedderError(f"ollama embed failed: {e}") from e
            if len(vecs) != len(batch):
                raise EmbedderError("ollama embed returned a wrong number of vectors")
            out.extend(vecs)
        return out

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"embedder": "ollama", "model": self.model}
        try:
            for m in self._client.get(f"{self.host}/api/tags").json().get("models", []):
                if m.get("name") in (self.model, f"{self.model}:latest"):
                    out["digest"] = m.get("digest")
                    out["details"] = m.get("details")
        except (httpx.HTTPError, ValueError) as e:
            out["info_error"] = str(e)
        return out


@register_embedder("hash")
class HashEmbedder(_Memo):
    """テスト・ドライラン用: トークンのハッシュで作る決定的なベクトル(意味はわからない)。"""

    name = "hash"

    def __init__(self, dim: int = 256, **_: Any) -> None:
        super().__init__()
        self.dim = dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
            out.append(v)
        return out

    def info(self) -> dict[str, Any]:
        return {"embedder": "hash", "dim": self.dim}
