"""Complete experiment pipeline runner for cell fate dynamics."""

from __future__ import annotations
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import scvi
import cellrank as cr

from cell_fate_dynamics.data import load_pancreas_data, preprocess_data
from cell_fate_dynamics.models import train_scvi, compute_velocity
from cell_fate_dynamics.evaluation import (
    evaluate_representation_clustering,
    neighborhood_overlap,
    evaluate_velocity_confidence,
)


def run_pipeline(output_dir: str = "figures", seed: int = 42) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    scvi.settings.seed = seed
    scv.settings.verbosity = 1
    sc.settings.verbosity = 1

    print("Step 1: Loading scVelo pancreas dataset...")
    adata = load_pancreas_data()
    print(f"Loaded dataset: {adata.n_obs} cells x {adata.n_vars} genes")

    # 1. Proportions figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scv.pl.proportions(adata, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_proportions_spliced_unspliced.png"), dpi=150)
    plt.close()

    print("Step 2: Preprocessing and baseline PCA...")
    adata.layers["counts"] = adata.X.copy()
    adata = preprocess_data(adata, min_shared_counts=20, n_top_genes=2000)

    # Compute UMAP on PCA
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, random_state=seed, key_added="neighbors_pca")
    sc.tl.umap(adata, neighbors_key="neighbors_pca", random_state=seed)
    adata.obsm["X_umap_pca"] = adata.obsm["X_umap"].copy()

    print("Step 3: Training scVI variational model...")
    adata, vae = train_scvi(
        adata,
        layer="counts",
        n_latent=30,
        max_epochs=45,
        batch_size=512,
        seed=seed,
        early_stopping=True,
        early_stopping_patience=8,
    )

    # Checkpoint saving
    os.makedirs("data", exist_ok=True)
    adata.write_h5ad("data/pancreas_processed.h5ad")
    vae.save("data/scvi_model", overwrite=True)
    print("Checkpoint saved to data/pancreas_processed.h5ad and data/scvi_model")

    # Compute UMAP on scVI latent representation
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, random_state=seed, key_added="neighbors_scvi")
    sc.tl.umap(adata, neighbors_key="neighbors_scvi", random_state=seed)
    adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()

    # Compare PCA vs scVI embeddings
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc.pl.embedding(adata, basis="umap_pca", color="clusters", ax=axes[0], show=False, title="PCA Representation UMAP")
    sc.pl.embedding(adata, basis="umap_scvi", color="clusters", ax=axes[1], show=False, title="scVI Latent Representation UMAP")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_pca_vs_scvi_umap.png"), dpi=150)
    plt.close()

    print("Step 4: Evaluating representation clustering & local geometry...")
    sil_scores = evaluate_representation_clustering(adata, cell_type_key="clusters", rep_keys=["X_pca", "X_scVI"])
    overlap_15 = neighborhood_overlap(adata.obsm["X_pca"], adata.obsm["X_scVI"], n_neighbors=15)
    overlap_30 = neighborhood_overlap(adata.obsm["X_pca"], adata.obsm["X_scVI"], n_neighbors=30)
    print(f"Silhouette scores: PCA={sil_scores.get('X_pca'):.4f}, scVI={sil_scores.get('X_scVI'):.4f}")
    print(f"Neighborhood overlap: k=15: {overlap_15:.4f}, k=30: {overlap_30:.4f}")

    print("Step 5: Estimating transition directions with scVelo...")
    adata = compute_velocity(adata, mode="stochastic", basis="umap_pca", n_jobs=-1)
    scv.tl.velocity_embedding(adata, basis="umap_scvi")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scv.pl.velocity_embedding_stream(adata, basis="umap_pca", color="clusters", ax=axes[0], show=False, title="scVelo Velocity (PCA UMAP)")
    scv.pl.velocity_embedding_stream(adata, basis="umap_scvi", color="clusters", ax=axes[1], show=False, title="scVelo Velocity (scVI UMAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_velocity_stream_pca_and_scvi.png"), dpi=150)
    plt.close()

    conf_metrics = evaluate_velocity_confidence(adata)

    print("Step 6: Terminal states with CellRank GPCCA...")
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix(n_jobs=-1, show_progress_bar=False)
    ck = cr.kernels.ConnectivityKernel(adata)
    ck.compute_transition_matrix()
    combined_kernel = 0.8 * vk + 0.2 * ck

    g = cr.estimators.GPCCA(combined_kernel)
    g.compute_macrostates(n_states=5, cluster_key="clusters")
    adata.obs["macrostates"] = g.macrostates

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.embedding(adata, basis="umap_pca", color="macrostates", ax=ax, show=False, title="CellRank Inferred Macrostates (GPCCA)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_cellrank_terminal_states.png"), dpi=150)
    plt.close()

    print("Step 7: Robustness & sensitivity evaluation...")
    robustness_results = {}
    rng = np.random.default_rng(seed)
    for frac in [0.9, 0.8, 0.7]:
        sub_idx = rng.choice(adata.n_vars, size=int(adata.n_vars * frac), replace=False)
        adata_sub = adata[:, sub_idx].copy()
        scv.tl.velocity(adata_sub, mode="stochastic")
        scv.tl.velocity_graph(adata_sub, n_jobs=-1)
        scv.tl.velocity_confidence(adata_sub)
        corr = float(np.corrcoef(adata.obs["velocity_confidence"].values, adata_sub.obs["velocity_confidence"].values)[0, 1])
        robustness_results[f"gene_downsample_{int(frac*100)}%"] = {
            "correlation_velocity_confidence": corr,
            "mean_confidence": float(adata_sub.obs["velocity_confidence"].mean()),
        }

    # Plot robustness comparison
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fractions = [70, 80, 90]
    corrs = [robustness_results[f"gene_downsample_{f}%"]["correlation_velocity_confidence"] for f in fractions]
    ax.plot(fractions, corrs, marker="o", color="#2b5c8f", lw=2, markersize=8)
    ax.set_xlabel("Gene Subsampling Percentage (%)", fontsize=11)
    ax.set_ylabel("Velocity Confidence Correlation with Full Dataset", fontsize=11)
    ax.set_title("Robustness of Velocity Inferences Under Gene Subsampling", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_robustness_perturbation_analysis.png"), dpi=150)
    plt.close()

    metrics = {
        "dataset": {
            "n_cells": int(adata.n_obs),
            "n_genes_raw": 27998,
            "n_genes_filtered": int(adata.n_vars),
            "cell_types": adata.obs["clusters"].value_counts().to_dict(),
        },
        "silhouette_scores": sil_scores,
        "neighborhood_overlap": {
            "jaccard_k15": float(overlap_15),
            "jaccard_k30": float(overlap_30),
        },
        "velocity_confidence": conf_metrics,
        "robustness": robustness_results,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Pipeline completed successfully! Metrics written to {metrics_path}")
    return metrics


if __name__ == "__main__":
    run_pipeline()
