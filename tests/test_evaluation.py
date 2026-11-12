"""Unit tests for evaluation and robustness metrics."""

import numpy as np
import pandas as pd
import anndata
import pytest
from cell_fate_dynamics.evaluation import (
    neighborhood_overlap,
    evaluate_representation_clustering,
    evaluate_velocity_confidence,
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
