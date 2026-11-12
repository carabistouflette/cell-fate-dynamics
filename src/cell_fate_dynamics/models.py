"""Model training and transition inference utilities."""

from __future__ import annotations
from typing import Optional, Literal
import anndata
import scvelo as scv


def train_scvi(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    n_latent: int = 30,
    max_epochs: int = 45,
    batch_size: int = 512,
    seed: int = 42,
    batch_key: Optional[str] = None,
    early_stopping: bool = True,
    early_stopping_patience: int = 8,
) -> tuple[anndata.AnnData, object]:
    """Train an scVI variational autoencoder to learn a latent cell representation.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    layer : Optional[str]
        Layer to use for training counts (default uses .X).
    n_latent : int
        Dimensionality of the latent representation.
    max_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size for dataloader.
    seed : int
        Random seed for reproducibility.
    batch_key : Optional[str]
        Optional batch covariate key in .obs.
    early_stopping : bool
        Whether to monitor validation loss and stop early.
    early_stopping_patience : int
        Epochs with no improvement before stopping.

    Returns
    -------
    tuple[anndata.AnnData, scvi.model.SCVI]
        Tuple of (AnnData with .obsm['X_scVI'] populated, trained model instance).
    """
    import scvi

    scvi.settings.seed = seed

    scvi.model.SCVI.setup_anndata(
        adata,
        layer=layer,
        batch_key=batch_key,
    )
    vae = scvi.model.SCVI(adata, n_latent=n_latent, n_layers=1)
    vae.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        plan_kwargs={"lr": 2e-3},
    )

    adata.obsm["X_scVI"] = vae.get_latent_representation()
    return adata, vae


def compute_velocity(
    adata: anndata.AnnData,
    mode: Literal["stochastic", "deterministic", "dynamical"] = "stochastic",
    basis: str = "umap",
    n_jobs: int = -1,
) -> anndata.AnnData:
    """Compute RNA velocity vectors using scVelo.

    Parameters
    ----------
    adata : anndata.AnnData
        Preprocessed AnnData with moments computed.
    mode : Literal["stochastic", "deterministic", "dynamical"]
        Velocity estimation model.
    basis : str
        Dimensionality reduction coordinates in .obsm (e.g. 'umap').
    n_jobs : int
        Number of parallel jobs for velocity graph.

    Returns
    -------
    anndata.AnnData
        AnnData with velocity layers and transition graph populated.
    """
    if mode == "dynamical":
        scv.tl.recover_dynamics(adata)

    scv.tl.velocity(adata, mode=mode)
    scv.tl.velocity_graph(adata, n_jobs=n_jobs)

    if f"X_{basis}" in adata.obsm:
        scv.tl.velocity_embedding(adata, basis=basis)

    return adata
