"""
Face-cluster visualization — classical MDS on the SearchFaces similarity
matrix captured during clustering (README.md — replaces the old t-SNE on
handcrafted quality features; zero extra API calls, identity-meaningful
layout: visually similar people sit near each other).
"""

from __future__ import annotations

import numpy as np


def mds_coords(similarity: np.ndarray) -> np.ndarray:
    """
    Classical MDS (Torgerson) from a symmetric similarity matrix in [0, 100]
    (Rekognition Similarity scores; diagonal = 100). Returns (n, 2) coords
    normalized to [-1, 1].
    """
    n = similarity.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)

    # Similarity → dissimilarity in [0, 1]
    d = 1.0 - np.clip(similarity, 0.0, 100.0) / 100.0
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)

    # Double centering: B = -1/2 J D² J
    d2 = d ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j

    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:2]
    top_vals = np.maximum(vals[order], 0.0)
    coords = vecs[:, order] * np.sqrt(top_vals)

    scale = np.abs(coords).max()
    if scale > 0:
        coords = coords / scale
    return coords.astype(np.float32)


def build_similarity_matrix(n: int, pairs: list[tuple[int, int, float]]) -> np.ndarray:
    """
    Assemble a symmetric matrix from SearchFaces match pairs
    (i, j, similarity 0-100). Unobserved pairs get 0 (max dissimilarity);
    diagonal is 100.
    """
    m = np.zeros((n, n), dtype=np.float32)
    for i, j, sim in pairs:
        m[i, j] = max(m[i, j], sim)
        m[j, i] = max(m[j, i], sim)
    np.fill_diagonal(m, 100.0)
    return m
