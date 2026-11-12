"""Quantitative evaluation and robustness check utilities."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import anndata
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import scvelo as scv


def evaluate_representation_clustering(
    adata: anndata.AnnData,
    cell_type_key: str = "clusters",
    rep_keys: Sequence[str] = ("X_pca", "X_scVI"),
) -> dict[str, float]:
    """Compute silhouette scores for cell-type separation across latent spaces.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated dataset with representation keys in .obsm.
    cell_type_key : str
        Column in adata.obs containing ground-truth cell annotations.
    rep_keys : Sequence[str]
        List of keys in adata.obsm to evaluate.

    Returns
    -------
    dict[str, float]
        Mapping of representation key to silhouette score.
    """
    labels = adata.obs[cell_type_key].values
    scores = {}

    for key in rep_keys:
        if key in adata.obsm:
            rep = adata.obsm[key]
            # If 2D or high-D, compute silhouette score on the representation
            score = silhouette_score(rep, labels)
            scores[key] = float(score)

    return scores


def neighborhood_overlap(
    rep1: np.ndarray,
    rep2: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    """Compute mean Jaccard overlap of k-nearest neighbors between two embeddings.

    Parameters
    ----------
    rep1 : np.ndarray
        First embedding matrix (N, D1).
    rep2 : np.ndarray
        Second embedding matrix (N, D2).
    n_neighbors : int
        Number of nearest neighbors to compare.

    Returns
    -------
    float
        Average Jaccard index across all cells in [0, 1].
    """
    nn1 = NearestNeighbors(n_neighbors=n_neighbors).fit(rep1)
    nn2 = NearestNeighbors(n_neighbors=n_neighbors).fit(rep2)

    # kneighbors() without query point already excludes the point itself
    indices1 = nn1.kneighbors(return_distance=False)
    indices2 = nn2.kneighbors(return_distance=False)

    jaccard_scores = []
    for idx1, idx2 in zip(indices1, indices2):
        s1 = set(idx1)
        s2 = set(idx2)
        union_len = len(s1.union(s2))
        jaccard = len(s1.intersection(s2)) / union_len if union_len > 0 else 0.0
        jaccard_scores.append(jaccard)

    return float(np.mean(jaccard_scores))


def evaluate_velocity_confidence(
    adata: anndata.AnnData,
) -> dict[str, float]:
    """Calculate average velocity confidence across cells.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with velocity computed.

    Returns
    -------
    dict[str, float]
        Velocity confidence and length statistics.
    """
    scv.tl.velocity_confidence(adata)
    metrics = {
        "mean_velocity_confidence": float(adata.obs["velocity_confidence"].mean()),
        "mean_velocity_length": float(adata.obs["velocity_length"].mean()),
    }
    return metrics


def validate_raw_counts(
    counts: np.ndarray | object,
    tolerance: float = 1e-5,
) -> bool:
    """Validate that count layer contains non-negative integer-like values.

    Parameters
    ----------
    counts : np.ndarray or sparse matrix
        Raw count matrix or layer.
    tolerance : float
        Allowed numerical tolerance from nearest integer.

    Returns
    -------
    bool
        True if all values are non-negative and integer-like.

    Raises
    ------
    ValueError
        If negative values or non-integer values are detected.
    """
    if hasattr(counts, "tocsr") or hasattr(counts, "data") and not isinstance(counts, np.ndarray):
        data = counts.data
    else:
        data = np.asarray(counts)
    if np.any(data < 0):
        raise ValueError("Count matrix contains negative values.")
    if not np.allclose(data, np.round(data), atol=tolerance):
        raise ValueError("Count matrix contains non-integer values (possibly normalized/log-transformed).")
    return True


def validate_fate_probabilities(
    fate_probabilities: np.ndarray | object,
    tolerance: float = 1e-3,
) -> bool:
    """Validate numerical properties of CellRank fate probabilities.

    Parameters
    ----------
    fate_probabilities : np.ndarray or Lineage object
        Cell x lineage transition probability matrix.
    tolerance : float
        Allowed deviation for row stochasticity (sum = 1.0).

    Returns
    -------
    bool
        True if all probabilities are finite, in [0, 1], and rows sum to 1.

    Raises
    ------
    ValueError
        If probabilities are invalid or row sums diverge from 1.
    """
    probs = np.asarray(fate_probabilities)
    if not np.all(np.isfinite(probs)):
        raise ValueError("Fate probabilities contain NaN or infinite values.")
    if np.any(probs < -tolerance) or np.any(probs > 1.0 + tolerance):
        raise ValueError("Fate probabilities are outside [0, 1].")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise ValueError("Fate probabilities rows do not sum to 1.0 within tolerance.")
    return True

