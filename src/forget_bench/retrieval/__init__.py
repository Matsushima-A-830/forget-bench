"""検索部品(埋め込み・BM25・順位融合)。rag / hybrid 戦略が使う。

検索器は回答バックエンドによらず共通(claude で回答する場合も、埋め込みはローカルの bge-m3)。
LLMの生成呼び出しではないので requests.jsonl には載せず、呼び出し数は meta.json の embedder_stats に記録する。
"""

from forget_bench.retrieval.bm25 import BM25Index, tokenize
from forget_bench.retrieval.embedder import Embedder, available_embedders, make_embedder
from forget_bench.retrieval.fusion import rrf

__all__ = ["BM25Index", "Embedder", "available_embedders", "make_embedder", "rrf", "tokenize"]
