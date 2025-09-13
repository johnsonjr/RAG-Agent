# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math
import re


def _tokenize(text: str) -> List[str]:
    # Simple word tokenizer: lowercase, alphanum tokens
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _tfidf_vectorize(docs: List[str]) -> Tuple[List[List[float]], List[str]]:
    # Build vocabulary
    vocab = sorted({tok for d in docs for tok in _tokenize(d)})
    df = {t: 0 for t in vocab}
    tokenized = []
    for d in docs:
        toks = _tokenize(d)
        tokenized.append(toks)
        seen = set(toks)
        for t in seen:
            df[t] += 1
    N = len(docs)
    vectors: List[List[float]] = []
    for toks in tokenized:
        tf = {t: 0 for t in vocab}
        for t in toks:
            tf[t] += 1
        vec = []
        for t in vocab:
            idf = math.log((N + 1) / (df[t] + 1)) + 1.0
            vec.append(tf[t] * idf)
        vectors.append(vec)
    return vectors, vocab


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SimpleRetriever:
    corpus: List[str]

    def __post_init__(self):
        self._vectors, self._vocab = _tfidf_vectorize(self.corpus)

    def query(self, q: str, k: int = 3) -> List[Tuple[str, float]]:
        q_vecs, _ = _tfidf_vectorize([q])
        qv = q_vecs[0]
        scores = [(_cosine(qv, dv), doc) for doc, dv in zip(self.corpus, self._vectors)]
        top = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        return [(doc, float(score)) for score, doc in top]
