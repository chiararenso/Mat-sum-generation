# Datasets

All data here is **public**. Confidential Akkodis data is never included.

## `paris/` — Paris-OSM 581 (prepared)
Semantically enriched, ready to run the Paris experiments (`MATSUM_OUT=data/paris`):

| file | content |
|---|---|
| `trajectories.parquet` | GPS points (581 individuals, 1.57 M points), one trajectory per de-identified OSM contributor |
| `semantic_locations.parquet` | dissolved semantic locations (MAT-Sum map-enrichment output) |
| `enriched_tiles.parquet` | per-tile OSM semantic context (labels / TF-IDF) |
| `tiles.parquet`, `summarized1/2.parquet` | tessellation and summarization intermediates |

`val_*.py`, `paris_*.py`, `unicity.py` read `trajectories`, `semantic_locations` and
`enriched_tiles` from here. Source: Human Mobility Datasets (Pugliese et al.), arXiv
2510.02333, Zenodo 15658129.

## `geolife/` — GeoLife Beijing 152 (prepared, partially)
Semantically enriched Beijing data, ready to run the GeoLife experiments
(`MATSUM_OUT=GEO_OUT=data/geolife`):

| file | content |
|---|---|
| `beijing.osm.pbf` | Beijing OSM extract (BBBike), input to `matsum_prepare.py` |
| `enriched_tiles.parquet`, `semantic_locations.parquet`, `tiles.parquet` | tessellation / OSM enrichment (MAT-Sum map-enrichment output, mirrors `paris/`) |
| `osm_features_raw.parquet` | cached raw OSM feature extraction (avoids re-parsing the pbf) |
| `geolife_sessions.parquet`, `geolife_cloud.parquet` | raw-audit intermediates (session index, downsampled point cloud) |
| `geolife_sess_cache.pkl` | per-session `(sig2, top1, user)` cache used by `geolife_e1..e4.py`, `geolife_disclosure.py`, `geolife_mia.py`, `geolife_release_size.py` |
| `geolife_E0.csv`, `geolife_release_size*.csv/png` | audit / release-size-sweep outputs |

**Not committed** (regenerable in a few minutes, hundreds of MB): `geolife_points.parquet`
(filtered point-level data) and `trajectories.parquet` (GeoDataFrame used by `matsum_prepare.py`).
To regenerate everything from scratch:

- **Download:** GeoLife GPS Trajectories 1.3 (182 users, Beijing) —
  https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
  (direct zip: `https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip`)
- **Beijing OSM** `.pbf`: already committed at `data/geolife/beijing.osm.pbf` (source: BBBike,
  `download.bbbike.org/osm/bbbike/Beijing/Beijing.osm.pbf` — re-download if a fresher OSM
  snapshot is needed; small natural drift vs. the paper's reported tile/label counts is expected
  since BBBike extracts update over time).
- **Prepare** (env: `GEOLIFE_DATA=.../Geolife Trajectories 1.3/Data`, `GEO_OUT=MATSUM_OUT=data/geolife`,
  `MATSUM_PBF=data/geolife/beijing.osm.pbf`, `MATSUM_INPUT=data/geolife/geolife_points.parquet`):
  run `geolife_audit.py` → `geolife_points.py` → `matsum_prepare.py` → `geolife_e0_semantic.py`
  (validate against the paper's Table 3/E0 numbers) → `geolife_build_cache.py` (writes
  `geolife_sess_cache.pkl`), then `geolife_e1..e4.py`, `geolife_disclosure.py`, `geolife_mia.py`,
  `geolife_release_size.py` as needed.

Source: Zheng et al., *GeoLife: A Collaborative Social Networking Service among User,
Location and Trajectory*, IEEE Data Eng. Bull., 2010.

## `labeled_OSM_category.csv`, `pois_categories_OSM.txt`
Public OSM tag → semantic-label dictionaries used by `matsum_prepare.py`.
