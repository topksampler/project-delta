"""Minimal BM25 Okapi — no external deps."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class BM25Index:
    """Serializable BM25 over a list of documents."""

    chunk_ids: list[str]
    doc_tokens: list[list[str]]
    doc_len: list[int]
    avgdl: float
    idf: dict[str, float]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> BM25Index:
        docs = [tokenize(t) for t in texts]
        n = len(docs)
        df: Counter[str] = Counter()
        for toks in docs:
            df.update(set(toks))
        idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        lengths = [len(toks) for toks in docs]
        avgdl = sum(lengths) / max(1, n)
        return cls(
            chunk_ids=list(chunk_ids),
            doc_tokens=docs,
            doc_len=lengths,
            avgdl=avgdl,
            idf=idf,
            k1=k1,
            b=b,
        )

    def score(self, query: str) -> list[tuple[str, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores: list[float] = [0.0] * len(self.doc_tokens)
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                freq = tf[term]
                idf = self.idf.get(term, 0.0)
                s += idf * (freq * (self.k1 + 1.0)) / (freq + denom_norm)
            scores[i] = s
        ranked = sorted(
            ((self.chunk_ids[i], scores[i]) for i in range(len(scores)) if scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def top_k(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        return self.score(query)[:k]

    def to_dict(self) -> dict:
        return {
            "chunk_ids": self.chunk_ids,
            "doc_tokens": self.doc_tokens,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BM25Index:
        return cls(
            chunk_ids=data["chunk_ids"],
            doc_tokens=data["doc_tokens"],
            doc_len=data["doc_len"],
            avgdl=float(data["avgdl"]),
            idf={k: float(v) for k, v in data["idf"].items()},
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )
