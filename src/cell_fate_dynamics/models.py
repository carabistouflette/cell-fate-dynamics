"""Model training and transition inference utilities."""

from __future__ import annotations
from typing import Optional, Literal
import anndata
import scvelo as scv


def train_scvi(
    adata: anndata.AnnData,
    layer: Optional[str] = None,
    n_latent: int = 30,
    max_epochs: int = 200,
    seed: int = 42,
    batch_key: Optional[str] = None,
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
    seed : int
        Random seed for reproducibility.
    batch_key : Optional[str]
        Optional batch covariate key in .obs.

    Returns
    -------
    tuple[anndata.AnnData, scvi.model.SCVI]
        Tuple of (AnnData with .obsm['X_scVI'] populated, trained model instance).
    """
    import scvi

    scvi.settings.seed = seed
    train_adata = adata.copy()

    scvi.model.SCVI.setup_anndata(
        train_adata,
        layer=layer,
        batch_key=batch_key,
    )
    vae = scvi.model.SCVI(train_adata, n_latent=n_latent)
    vae.train(max_epochs=max_epochs)

    adata.obsm["X_scVI"] = vae.get_latent_representation(adata)
    return adata, vae


def compute_velocity(
    adata: anndata.AnnData,
    mode: Literal["stochastic", "deterministic", "dynamical"] = "stochastic",
    basis: str = "umap",
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

    Returns
    -------
    anndata.AnnData
        AnnData with velocity layers and transition graph populated.
    """
    if mode == "dynamical":
        scv.tl.recover_dynamics(adata)

    scv.tl.velocity(adata, mode=mode)
    scv.tl.velocity_graph(adata)

    if f"X_{basis}" in adata.obsm:
        scv.tl.velocity_embedding(adata, basis=basis)

    return adata
