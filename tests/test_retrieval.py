from __future__ import annotations

import httpx
import pytest

from forget_bench.retrieval import BM25Index, make_embedder, rrf, tokenize
from forget_bench.retrieval.embedder import EmbedderError, OllamaEmbedder


def test_tokenize_mixed() -> None:
    assert tokenize("札幌に住む") == ["札幌", "幌に", "に住", "住む"]
    assert tokenize("My phone is 090-1234") == ["my", "phone", "is", "090", "1234"]
    assert "pp6430089" in tokenize("パスポートはPP6430089です")


def test_bm25_ranks_and_removes() -> None:
    idx = BM25Index()
    idx.add("a", "私は札幌に住んでいます")
    idx.add("b", "予算は350万円です")
    idx.add("c", "今日は雨ですね")
    s = idx.scores("住んでいる都市はどこ")
    assert max(s, key=s.get) == "a"
    idx.remove("a")
    assert "a" not in idx.scores("札幌")
    assert all(v >= 1 for v in idx.df.values())


def test_rrf() -> None:
    assert rrf([["a", "b", "c"], ["c", "a", "b"]]) == ["a", "c", "b"]
    assert rrf([["x"], []]) == ["x"]


def test_hash_embedder_memo() -> None:
    e = make_embedder("hash")
    v1 = e.embed(["abc", "abc", "def"])
    assert v1[0] == v1[1]
    assert e.stats() == {"calls": 1, "texts": 2, "memo_hits": 1}
    e.embed(["abc"])
    assert e.stats()["calls"] == 1 and e.stats()["memo_hits"] == 2


def test_ollama_embedder_batches_and_normalizes() -> None:
    seen: list[list[str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(req.content)
        seen.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    e = OllamaEmbedder(batch_size=2, client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = e.embed(["a", "b", "c"])
    assert seen == [["a", "b"], ["c"]]
    assert out[0] == pytest.approx([0.6, 0.8])


def test_ollama_embedder_error() -> None:
    e = OllamaEmbedder(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(EmbedderError):
        e.embed(["a"])
