"""BM25(Okapi)。文書の追加・削除ができる小さな実装。

トークン化: 正規化(NFKC・小文字化・記号除去)した文字列を、ASCII英数字の連続は単語、
それ以外(日本語など)は文字バイグラムに分ける。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

_ASCII_RUN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for chunk in re.split(r"[\s\W_]+", text):
        if not chunk:
            continue
        pos = 0
        for m in _ASCII_RUN.finditer(chunk):
            tokens.extend(_bigrams(chunk[pos : m.start()]))
            tokens.append(m.group(0))
            pos = m.end()
        tokens.extend(_bigrams(chunk[pos:]))
    return tokens


def _bigrams(s: str) -> list[str]:
    if not s:
        return []
    if len(s) == 1:
        return [s]
    return [s[i : i + 2] for i in range(len(s) - 1)]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.df: Counter[str] = Counter()

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.docs:
            self.remove(doc_id)
        tf = Counter(tokenize(text))
        self.docs[doc_id] = tf
        self.lengths[doc_id] = sum(tf.values())
        self.df.update(tf.keys())

    def remove(self, doc_id: str) -> None:
        tf = self.docs.pop(doc_id, None)
        if tf is None:
            return
        self.lengths.pop(doc_id)
        self.df.subtract(tf.keys())
        self.df += Counter()  # 0 以下を消す

    def scores(self, query: str) -> dict[str, float]:
        n = len(self.docs)
        if n == 0:
            return {}
        avg = sum(self.lengths.values()) / n or 1.0
        q = set(tokenize(query))
        out: dict[str, float] = {}
        for doc_id, tf in self.docs.items():
            s = 0.0
            dl = self.lengths[doc_id]
            for tok in q:
                f = tf.get(tok, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[tok] + 0.5) / (self.df[tok] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
            out[doc_id] = s
        return out
