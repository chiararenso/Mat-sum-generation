# Semantic Abstraction for Privacy-Aware Synthetic Mobility Generation

Code and results for the paper *"Semantic Abstraction for Privacy-Aware Synthetic
Mobility Generation"* (ISTI-CNR, Pisa). The project repurposes **MAT-Sum** (Pugliese,
Lettich, Pinelli, Renso — *Summarizing Trajectories Using Semantically Enriched
Geographical Context*, SIGSPATIAL '23) as the **generative substrate** for synthetic
mobility data: raw trajectories are abstracted into sequences of semantic locations, a
minimum-support semantic vocabulary is built, and a sequence generator (order-1 Markov or
a light LSTM) is learned over that abstract space. We evaluate the role of the
representation, small-data scaling and memorization, the privacy–fidelity trade-off of
generator capacity, and the privacy/utility of the released synthetic data, on two public
datasets (**Paris-OSM 581** and **GeoLife**).

> **Data note.** This repository contains **only code, documentation and results derived
> from public datasets**. The confidential **Akkodis** GPS data used in the original
> proof-of-concept (and every artifact derived from it) is intentionally **excluded** —
> see `.gitignore`. It is not redistributable.

## Repository layout

```
.
├── README.md                 # this file
├── requirements.txt          # Python 3.12 dependencies
├── paper/                    # LaTeX + Markdown draft of the paper
├── results/
│   ├── paris/                # Paris-581 result CSVs (summaries + raw replicate data)
│   └── geolife/              # GeoLife result CSVs
├── figures/                  # figures used in the paper (public-data results only)
└── *.py                      # all pipeline / experiment scripts (see map below)
```

Scripts are kept flat at the top level because they import one another as siblings
(`import matsum_summarize`, `from val_rq3 import ...`); the logical grouping is documented
below rather than enforced by directories.

## Environment

No system-wide scientific stack is assumed. Create a virtual environment and install the
pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is only needed for the LSTM generator (RQ3 / E3). `pyrosm` is only needed to
(re)build the semantic map from an OSM `.pbf` extract.

## Data provenance (all public)

| Dataset | Source |
|---|---|
| Paris-OSM 581 | Human Mobility Datasets (Pugliese et al.), arXiv **2510.02333**, Zenodo **15658129**, `github.com/Fr4nz83/MAT-Dataset` |
| GeoLife | Microsoft Research GeoLife 1.3 (182 users, Beijing) |
| Île-de-France OSM `.pbf` | Geofabrik |
| Beijing OSM `.pbf` | BBBike |

Large/regenerable intermediates (`*.parquet`, `.pbf`, GeoLife working dir) are **not**
committed; they are reproduced from the sources above by the preparation scripts. The
pipeline is dataset-agnostic and driven by environment variables:

```bash
export MATSUM_INPUT=/path/to/raw_trajectories.parquet   # or .csv
export MATSUM_OUT=/path/to/output                        # prepared parquets + result CSVs
export MATSUM_FIG=/path/to/figures                        # saved figures
export MATSUM_PBF=/path/to/region.osm.pbf                 # map enrichment (prepare only)
export GEO_OUT=/path/to/geolife                           # GeoLife working dir
```

## Script map (→ paper section)

### Core MAT-Sum pipeline & generators
| File | Role |
|---|---|
| `matsum_prepare.py`   | Stage 1 — tessellation, OSM aspect enrichment, semantic locations |
| `matsum_summarize.py` | Stage 2 — semantic mapping (temporal weights), summarization, S_rate |
| `matsum_muitas.py`    | MUITAS similarity + S_rate/MUITAS trade-off |
| `matsum_synth.py` / `matsum_synth_sweep.py` | semantic order-1 Markov generation; (m,k) vocabulary sweep |
| `matsum_deep.py`      | LSTM (higher-capacity) generator |
| `matsum_spatial.py` / `matsum_coupled.py` | decoupled vs spatially-coupled EPR instantiation (spatial layer) |
| `matsum_visualize.py` / `matsum_figures.py` | interactive maps / figures |

### Paris-581 — main validation (§4, RQ1–RQ3)
| File | Paper |
|---|---|
| `val_paris_eval.py`   | vectorized MUITAS evaluation at scale |
| `val_abstraction.py`, `val_abstraction_ci.py` | **RQ1** — role of the abstraction (grid / cluster / MAT-Sum at matched state count) |
| `val_rq2.py`          | **RQ2** — small-data scaling & memorization |
| `val_rq3.py`          | **RQ3** — generator capacity & privacy–fidelity (Markov vs LSTM; bigram+trigram TV) |
| `val_scaling.py`, `val_deep_scaling.py`, `val_altmodels.py` | supporting scaling / VOMM / DP-Markov studies |

### GeoLife — external validation (§4.5–4.8)
| File | Paper |
|---|---|
| `geolife_audit.py`, `geolife_points.py`, `geolife_e0_semantic.py`, `geolife_retention.py` | **E0** — study-area audit, frozen filters, retention |
| `geolife_e1.py` | **E1** — role of the abstraction |
| `geolife_e2.py` | **E2** — scaling (with saturation-point estimate) |
| `geolife_e3.py` | **E3** — generator capacity (Markov vs LSTM) |
| `geolife_e4.py` | **E4** — normalized Paris↔GeoLife comparison |

### Privacy & utility of the released data (§5)
| File | Paper |
|---|---|
| `paris_tstr.py` | **Utility** — TSTR next-semantic-label prediction |
| `paris_disclosure.py`, `geolife_disclosure.py`, `replot_disclosure.py` | **Disclosure tail** — near-copy vs threshold sweep + DCR tail, vs a real-to-real baseline |
| `paris_mia.py`, `geolife_mia.py` | **Membership inference** — distance-to-closest-record attack (ROC-AUC, TPR@10%FPR) |
| `unicity.py` | **Unicity** — de Montjoye-style semantic re-identification (real vs synthetic) |

`build_report.py`, `build_results.py`, `build_val_report.py` generate HTML reports (not
committed; may embed non-public data).

## Reproducing a result

Example — RQ2 on Paris (needs the prepared Paris outputs in `MATSUM_OUT`):

```bash
export MATSUM_OUT=/path/to/paris/output MATSUM_FIG=/path/to/figures
python val_rq2.py        # writes val_rq2_summary.csv + fig_rq2.png
```

Each experiment script prints a summary table and writes its CSV(s) to `MATSUM_OUT` and its
figure to `MATSUM_FIG`. The committed `results/` CSVs are the exact numbers reported in the
paper.

## Method choices vs the reference MAT-Sum repo

Faithful re-implementation of `github.com/chiarap2/MAT-Sum`; differences are plumbing only:

| Reference repo | Here | Why |
|---|---|---|
| `tesspy` tessellation | `mercantile` quadkey grid | same square grid at a zoom level, without the heavy dependency |
| live Overpass / `osmnx` | offline `.pbf` via `pyrosm` | deterministic, reproducible (public servers were unreliable) |
| config across many JSON files | one config + env vars | reproducibility |

The MAT-Sum algorithm itself (tessellation → OSM enrichment → TF-IDF labels → semantic
locations → semantic mapping → summarization → S_rate) is unchanged.
