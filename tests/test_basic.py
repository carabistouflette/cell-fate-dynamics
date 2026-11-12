"""Basic smoke tests for cell_fate_dynamics."""

def test_imports():
    """Ensure modules can be imported without errors."""
    import cell_fate_dynamics
    assert hasattr(cell_fate_dynamics, "load_pancreas_data")
    assert hasattr(cell_fate_dynamics, "preprocess_data")
    assert hasattr(cell_fate_dynamics, "train_scvi")
    assert hasattr(cell_fate_dynamics, "compute_velocity")
    assert hasattr(cell_fate_dynamics, "evaluate_representation_clustering")
    assert hasattr(cell_fate_dynamics, "neighborhood_overlap")


def test_neighborhood_overlap_identical():
    """Identical embeddings must have overlap 1.0."""
    import numpy as np
    from cell_fate_dynamics.evaluation import neighborhood_overlap

    x = np.random.randn(50, 5)
    overlap = neighborhood_overlap(x, x, n_neighbors=10)
    assert np.isclose(overlap, 1.0)

