"""Data loading and preprocessing utilities for single-cell fate dynamics."""

from __future__ import annotations
from typing import Optional
import anndata
import scanpy as sc
import scvelo as scv


def load_pancreas_data(file_path: Optional[str] = None) -> anndata.AnnData:
    """Load the public scVelo pancreas dataset.

    Parameters
    ----------
    file_path : Optional[str]
        Optional local file path to load from or cache to. If None, fetches the
        standard scVelo pancreas dataset.

    Returns
    -------
    anndata.AnnData
        Annotated data object containing spliced/unspliced counts and cell annotations.
    """
    if file_path is not None:
        return sc.read(file_path)
    return scv.datasets.pancreas()


def preprocess_data(
    adata: anndata.AnnData,
    min_shared_counts: int = 20,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    copy: bool = True,
) -> anndata.AnnData:
    """Run standard quality filtering, normalization, and neighborhood graph computation.

    Parameters
    ----------
    adata : anndata.AnnData
        Raw or minimally processed AnnData with spliced and unspliced layers.
    min_shared_counts : int
        Minimum count threshold for spliced/unspliced sharing.
    n_top_genes : int
        Number of highly variable genes to select.
    n_pcs : int
        Number of principal components for neighbor graph.
    n_neighbors : int
        Number of neighbors for kNN graph.
    copy : bool
        Whether to return a copy or modify inplace.

    Returns
    -------
    anndata.AnnData
        Preprocessed AnnData object ready for velocity and representation learning.
    """
    if copy:
        adata = adata.copy()

    # scVelo filtering and normalization followed by log1p and HVG selection
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    return adata
