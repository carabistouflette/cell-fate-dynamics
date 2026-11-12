"""Unit tests for evaluation and robustness metrics."""

import numpy as np
import pandas as pd
import anndata
import pytest
from cell_fate_dynamics.evaluation import (
    neighborhood_overlap,
    evaluate_representation_clustering,
    evaluate_velocity_confidence,
    validate_raw_counts,
    validate_fate_probabilities,
)


def test_neighborhood_overlap_bounds():
    """Jaccard overlap must be bounded between 0 and 1."""
    rng = np.random.default_rng(42)
    rep1 = rng.normal(size=(60, 5))
    rep2 = rng.normal(size=(60, 5))

    overlap = neighborhood_overlap(rep1, rep2, n_neighbors=10)
    assert 0.0 <= overlap <= 1.0


def test_neighborhood_overlap_identical():
    """Identical embeddings must yield overlap 1.0."""
    rng = np.random.default_rng(123)
    rep = rng.normal(size=(40, 4))
    overlap = neighborhood_overlap(rep, rep, n_neighbors=8)
    assert np.isclose(overlap, 1.0)


def test_evaluate_representation_clustering():
    """Silhouette scores must correctly reflect well-separated vs random clusters."""
    rng = np.random.default_rng(999)
    # Well separated clusters
    c1 = rng.normal(loc=-5.0, scale=0.5, size=(30, 2))
    c2 = rng.normal(loc=5.0, scale=0.5, size=(30, 2))
    separated = np.vstack([c1, c2])

    # Overlapping random noise
    random_rep = rng.normal(scale=1.0, size=(60, 2))

    labels = ["A"] * 30 + ["B"] * 30
    adata = anndata.AnnData(
        X=separated,
        obs=pd.DataFrame({"cell_type": labels}),
        obsm={"X_separated": separated, "X_random": random_rep},
    )

    scores = evaluate_representation_clustering(
        adata,
        cell_type_key="cell_type",
        rep_keys=["X_separated", "X_random"],
    )

    assert "X_separated" in scores
    assert "X_random" in scores
    assert scores["X_separated"] > scores["X_random"]
    assert scores["X_separated"] > 0.7


def test_evaluate_representation_clustering_missing_key():
    """Handles missing representation keys gracefully."""
    adata = anndata.AnnData(
        X=np.zeros((10, 2)),
        obs=pd.DataFrame({"cell_type": ["A"] * 5 + ["B"] * 5}),
        obsm={"X_existing": np.random.randn(10, 2)},
    )
    scores = evaluate_representation_clustering(
        adata,
        cell_type_key="cell_type",
        rep_keys=["X_existing", "X_nonexistent"],
    )
    assert "X_existing" in scores
    assert "X_nonexistent" not in scores


def test_validate_raw_counts_success():
    """Validates non-negative integer count matrix."""
    import scipy.sparse as sp
    counts_dense = np.array([[0, 5, 12], [3, 0, 8]], dtype=float)
    assert validate_raw_counts(counts_dense) is True

    counts_sparse = sp.csr_matrix(counts_dense)
    assert validate_raw_counts(counts_sparse) is True


def test_validate_raw_counts_negative():
    """Rejects count matrix with negative values."""
    negative_counts = np.array([[0, -1], [2, 3]], dtype=float)
    with pytest.raises(ValueError, match="negative values"):
        validate_raw_counts(negative_counts)


def test_validate_raw_counts_non_integer():
    """Rejects normalized/float-transformed counts."""
    float_counts = np.array([[0.5, 1.2], [2.0, 3.1]])
    with pytest.raises(ValueError, match="non-integer"):
        validate_raw_counts(float_counts)


def test_validate_fate_probabilities_success():
    """Validates proper row-stochastic probability matrix."""
    probs = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [1.0, 0.0, 0.0]])
    assert validate_fate_probabilities(probs) is True


def test_validate_fate_probabilities_non_sum_to_one():
    """Rejects fate probabilities whose rows do not sum to 1."""
    probs = np.array([[0.5, 0.2], [0.3, 0.3]])
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_fate_probabilities(probs)


def test_validate_fate_probabilities_out_of_bounds():
    """Rejects probabilities < 0 or > 1."""
    probs = np.array([[1.2, -0.2], [0.5, 0.5]])
    with pytest.raises(ValueError, match="outside"):
        validate_fate_probabilities(probs)

