"""Rank embedding search results by cosine similarity, not raw dot product.

The help-centre search embedded articles and queries with a model whose
vectors are not unit length, and the vector index scored them with the
inner (dot) product. A dot product grows with vector length as well as with
direction, and the long "Terms of Service" article had by far the longest
vector, so it came first for almost every query.

The fix L2-normalises every vector when it is indexed and every query when it
is searched, so the dot product becomes cosine similarity and only the
direction (the meaning) counts. Zero vectors, which have no direction, are
rejected instead of silently scoring 0 or dividing by zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vector = list[float]

# Vectors shorter than this are treated as zero: they carry no direction.
MIN_VECTOR_NORM = 1e-12


class ZeroVectorError(ValueError):
    """A vector with (almost) zero length cannot be normalised."""


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float


def dot(left: Vector, right: Vector) -> float:
    """Inner product of two vectors of the same dimension."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}")
    total = 0.0
    for left_value, right_value in zip(left, right):
        total += left_value * right_value
    return total


def l2_normalize(vector: Vector) -> Vector:
    """Scale a vector to length 1 so only its direction is left."""
    norm = math.sqrt(dot(vector, vector))
    if norm < MIN_VECTOR_NORM:
        raise ZeroVectorError("cannot normalise a zero vector (empty or failed embedding?)")
    return [value / norm for value in vector]


def top_k(scored: list[SearchHit], k: int) -> list[SearchHit]:
    """Highest scores first; equal scores ordered by doc id so results are stable."""
    ordered = sorted(scored, key=lambda hit: (-hit.score, hit.doc_id))
    return ordered[:k]


def naive_search(documents: dict[str, Vector], query: Vector, k: int) -> list[SearchHit]:
    """The original index: raw dot product, so long vectors win."""
    scored = [SearchHit(doc_id, dot(vector, query)) for doc_id, vector in documents.items()]
    return top_k(scored, k)


class CosineIndex:
    """Vector index that stores unit vectors, so dot product == cosine similarity."""

    def __init__(self) -> None:
        self._unit_vectors: dict[str, Vector] = {}

    def add(self, doc_id: str, vector: Vector) -> None:
        """Normalise once at write time, so search stays a plain dot product."""
        self._unit_vectors[doc_id] = l2_normalize(vector)

    def search(self, query: Vector, k: int) -> list[SearchHit]:
        """The k documents pointing most in the query's direction."""
        unit_query = l2_normalize(query)
        scored = []
        for doc_id, unit_vector in self._unit_vectors.items():
            scored.append(SearchHit(doc_id, dot(unit_vector, unit_query)))
        return top_k(scored, k)


def top_result_counts(results_per_query: list[list[SearchHit]]) -> dict[str, int]:
    """How many queries each document was the first result for."""
    counts: dict[str, int] = {}
    for hits in results_per_query:
        if hits:
            first = hits[0].doc_id
            counts[first] = counts.get(first, 0) + 1
    return counts
