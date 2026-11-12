# Cell Fate Dynamics

*A small, reproducible exploration of cell-state transitions in single-cell RNA-seq data.*

## Why this project?

Single-cell RNA sequencing captures cells at different states, but usually not their full trajectories over time. This proof of concept explores what can be learned about cell differentiation from these noisy snapshots — and where the limits of such inferences lie.

The goal is **not** to claim a complete reconstruction of the underlying biological system. It is to build, compare, and critically evaluate a few accessible methods.

## Dataset

We use the public [pancreas dataset provided by scVelo](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.pancreas.html), which contains cells at different stages of endocrine differentiation, including spliced and unspliced RNA measurements.

## Approach

1. **Explore and preprocess** the expression data; inspect cell types and basic quality metrics.
2. **Learn low-dimensional representations** with PCA (baseline) and a variational model such as [scVI](https://docs.scvi-tools.org/).
3. **Estimate transition directions** using [scVelo](https://scvelo.readthedocs.io/) and, if feasible, terminal-state probabilities with [CellRank](https://cellrank.readthedocs.io/).
4. **Evaluate the results** rather than relying on attractive visualizations alone.

A latent representation and an inferred transition model are distinct components; a 2D embedding alone does not establish cell dynamics.

## Questions to investigate

- Do PCA and scVI preserve known cell populations and intermediate states?
- Do inferred transitions follow biologically plausible differentiation paths?
- How stable are the conclusions across random seeds, selected genes, or added noise?
- Which findings can be interpreted, and which remain uncertain?

## Intended outputs

- One self-contained, runnable notebook (ideally on Google Colab).
- Clear figures comparing representations and inferred transitions.
- A small quantitative comparison, including at least one robustness check.
- A short discussion of limitations and unexpected or negative results.

## Project status

**Proof of concept — in development.** Methods, figures, and conclusions will be added as experiments are completed. No biological or predictive claims are made at this stage.

## Reproducibility

The project will document data sources, package versions, random seeds, and execution steps. All analyses use publicly available data.
