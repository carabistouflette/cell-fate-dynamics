"""Complete experiment pipeline runner for cell fate dynamics."""

from __future__ import annotations
import argparse
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
    validate_raw_counts,
    validate_fate_probabilities,
)


def run_pipeline(
    output_dir: str = "figures",
    seed: int = 42,
    use_cache: bool = False,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    scvi.settings.seed = seed
    scv.settings.verbosity = 1
    sc.settings.verbosity = 1

    cache_file = "data/pancreas_processed.h5ad"
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached preprocessed dataset from {cache_file}...")
        adata = sc.read_h5ad(cache_file)
    else:
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
        validate_raw_counts(adata.X)
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

        # Compute UMAP on scVI latent representation
        sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30, random_state=seed, key_added="neighbors_scvi")
        sc.tl.umap(adata, neighbors_key="neighbors_scvi", random_state=seed)
        adata.obsm["X_umap_scvi"] = adata.obsm["X_umap"].copy()

        # Checkpoint saving with both UMAPs included
        os.makedirs("data", exist_ok=True)
        adata.write_h5ad(cache_file)
        vae.save("data/scvi_model", overwrite=True)
        print(f"Checkpoint saved to {cache_file} and data/scvi_model")

    # Ensure UMAPs exist
    if "X_umap_pca" not in adata.obsm:
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, random_state=seed, key_added="neighbors_pca")
        sc.tl.umap(adata, neighbors_key="neighbors_pca", random_state=seed)
        adata.obsm["X_umap_pca"] = adata.obsm["X_umap"].copy()

    if "X_umap_scvi" not in adata.obsm:
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
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap_pca",
        color="clusters",
        ax=axes[0],
        show=False,
        title="scVelo Velocity (PCA UMAP projection)",
    )
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap_scvi",
        color="clusters",
        ax=axes[1],
        show=False,
        title="scVelo Velocity (scVI UMAP projection)\n[Note: same velocity model projected on different embedding]",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_velocity_stream_pca_and_scvi.png"), dpi=150)
    plt.close()

    conf_metrics = evaluate_velocity_confidence(adata)

    print("Step 6: Predicting terminal states with CellRank GPCCA...")
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix(n_jobs=-1, show_progress_bar=False)
    ck = cr.kernels.ConnectivityKernel(adata)
    ck.compute_transition_matrix()
    combined_kernel = 0.8 * vk + 0.2 * ck

    g = cr.estimators.GPCCA(combined_kernel)
    g.compute_macrostates(n_states=5, cluster_key="clusters")
    adata.obs["macrostates"] = g.macrostates

    # Predict terminal states based on macrostate stability
    g.predict_terminal_states(method="stability")
    adata.obs["terminal_states"] = g.terminal_states
    predicted_terminals = [str(c) for c in g.terminal_states.cat.categories]
    print(f"Predicted terminal states: {predicted_terminals}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.embedding(
        adata,
        basis="umap_pca",
        color="terminal_states",
        ax=ax,
        show=False,
        title="CellRank Predicted Terminal States (GPCCA Stability)",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_cellrank_terminal_states.png"), dpi=150)
    plt.close()

    # Step 6b: Compute fate probabilities
    print("Step 6b: Computing fate probabilities...")
    g.compute_fate_probabilities(solver="gmres", use_petsc=False)
    validate_fate_probabilities(g.fate_probabilities)

    terminal_lineages = list(g.fate_probabilities.names)
    print(f"Terminal lineages: {terminal_lineages}")

    n_lineages = len(terminal_lineages)
    fig, axes = plt.subplots(1, n_lineages, figsize=(4.8 * n_lineages, 4.2))
    if n_lineages == 1:
        axes = [axes]

    for idx, lineage in enumerate(terminal_lineages):
        col_name = f"fate_{lineage}"
        adata.obs[col_name] = np.array(g.fate_probabilities[:, lineage]).flatten()
        sc.pl.embedding(
            adata,
            basis="umap_pca",
            color=col_name,
            cmap="viridis",
            ax=axes[idx],
            show=False,
            title=f"Fate Prob -> {lineage}",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_fate_probabilities.png"), dpi=150)
    plt.close()

    # Step 6c: Identify lineage drivers
    print("Step 6c: Identifying lineage drivers...")
    df_drivers = g.compute_lineage_drivers(cluster_key="clusters")

    top_lineage_drivers = {}
    fig, axes = plt.subplots(1, n_lineages, figsize=(4.4 * n_lineages, 4.8))
    if n_lineages == 1:
        axes = [axes]

    for idx, lineage in enumerate(terminal_lineages):
        corr_col = f"{lineage}_corr"
        pval_col = f"{lineage}_pval"
        if corr_col in df_drivers.columns:
            top_5 = df_drivers.sort_values(by=corr_col, ascending=False).head(5)
            top_lineage_drivers[lineage] = [
                {
                    "gene": str(gene),
                    "correlation": float(row[corr_col]),
                    "pval": float(row[pval_col]),
                }
                for gene, row in top_5.iterrows()
            ]

            genes = list(top_5.index)[::-1]
            corrs = [top_5.loc[g_name, corr_col] for g_name in genes]
            axes[idx].barh(genes, corrs, color="#2b5c8f", alpha=0.85)
            axes[idx].set_xlabel("Pearson Correlation", fontsize=9)
            axes[idx].set_title(f"Drivers: {lineage}", fontsize=11)
            axes[idx].set_xlim(0, 1.0)
            axes[idx].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_lineage_drivers.png"), dpi=150)
    plt.close()

    print("Step 7: Robustness & sensitivity evaluation...")
    robustness_results = {}
    rng = np.random.default_rng(seed)

    full_fate_probs = {lineage: np.array(g.fate_probabilities[:, lineage]).flatten() for lineage in terminal_lineages}

    for frac in [0.9, 0.8, 0.7]:
        sub_idx = rng.choice(adata.n_vars, size=int(adata.n_vars * frac), replace=False)
        adata_sub = adata[:, sub_idx].copy()
        scv.tl.velocity(adata_sub, mode="stochastic")
        scv.tl.velocity_graph(adata_sub, n_jobs=-1)
        scv.tl.velocity_confidence(adata_sub)
        corr_conf = float(np.corrcoef(adata.obs["velocity_confidence"].values, adata_sub.obs["velocity_confidence"].values)[0, 1])

        # CellRank transition kernel on downsampled data to check fate stability
        fate_corrs = {}
        try:
            vk_sub = cr.kernels.VelocityKernel(adata_sub)
            vk_sub.compute_transition_matrix(n_jobs=-1, show_progress_bar=False)
            ck_sub = cr.kernels.ConnectivityKernel(adata_sub)
            ck_sub.compute_transition_matrix()
            g_sub = cr.estimators.GPCCA(0.8 * vk_sub + 0.2 * ck_sub)
            g_sub.compute_macrostates(n_states=5, cluster_key="clusters")
            g_sub.set_terminal_states(states=terminal_lineages)
            g_sub.compute_fate_probabilities(solver="gmres", use_petsc=False)
            for lineage in terminal_lineages:
                sub_prob = np.array(g_sub.fate_probabilities[:, lineage]).flatten()
                r = float(np.corrcoef(full_fate_probs[lineage], sub_prob)[0, 1])
                fate_corrs[lineage] = r
        except Exception as e:
            print(f"Notice: Subset GPCCA fate calculation skipped for frac {frac}: {e}")

        robustness_results[f"gene_downsample_{int(frac*100)}%"] = {
            "correlation_velocity_confidence": corr_conf,
            "mean_confidence": float(adata_sub.obs["velocity_confidence"].mean()),
            "fate_probabilities_correlation": fate_corrs,
        }

    # Plot two-panel robustness figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fractions = [70, 80, 90]
    corrs_conf = [robustness_results[f"gene_downsample_{f}%"]["correlation_velocity_confidence"] for f in fractions]
    axes[0].plot(fractions, corrs_conf, marker="o", color="#2b5c8f", lw=2, markersize=8)
    axes[0].set_xlabel("Gene Subsampling Percentage (%)", fontsize=11)
    axes[0].set_ylabel("Correlation with Full Dataset", fontsize=11)
    axes[0].set_title("RNA Velocity Confidence Stability", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot fate probability stability across lineages
    markers = ["s", "^", "D", "v", "o"]
    has_fate_plots = False
    for i, lineage in enumerate(terminal_lineages):
        lineage_corrs = [
            robustness_results[f"gene_downsample_{f}%"].get("fate_probabilities_correlation", {}).get(lineage, np.nan)
            for f in fractions
        ]
        if not all(np.isnan(lineage_corrs)):
            axes[1].plot(fractions, lineage_corrs, marker=markers[i % len(markers)], lw=2, label=f"Fate -> {lineage}")
            has_fate_plots = True

    axes[1].set_xlabel("Gene Subsampling Percentage (%)", fontsize=11)
    axes[1].set_ylabel("Correlation with Full Dataset", fontsize=11)
    axes[1].set_title("CellRank Terminal Fate Stability", fontsize=12)
    axes[1].set_ylim(0, 1.05)
    if has_fate_plots:
        axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

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
        "predicted_terminal_states": predicted_terminals,
        "lineage_drivers": top_lineage_drivers,
        "robustness": robustness_results,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Pipeline completed successfully! Metrics written to {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell Fate Dynamics experiment pipeline runner.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Re-use cached preprocessed data and scVI latent space (data/pancreas_processed.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save figures and metrics (default: figures).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        output_dir=args.output_dir,
        seed=args.seed,
        use_cache=args.use_cache,
    )
