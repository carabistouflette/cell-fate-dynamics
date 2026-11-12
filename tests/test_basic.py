"""Basic smoke tests for cell_fate_dynamics."""

def test_imports():
    """Ensure modules can be imported without errors."""
    import cell_fate_dynamics
    assert hasattr(cell_fate_dynamics, "load_pancreas_data")
    assert hasattr(cell_fate_dynamics, "preprocess_data")
    assert hasattr(cell_fate_dynamics, "train_scvi")
    assert hasattr(cell_fate_dynamics, "compute_velocity")
    assert hasattr(cell_fate_dynamics, "evaluate_representation_clustering")
