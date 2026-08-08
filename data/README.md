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

## GeoLife (not stored here — download from Microsoft Research)
GeoLife is a large, well-known public dataset, so it is **not committed**. Download it and
regenerate the prepared artifacts locally:

- **Download:** GeoLife GPS Trajectories 1.3 (182 users, Beijing) —
  https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
  (direct zip: `https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip`)
- **Beijing OSM** `.pbf`: BBBike (`download.bbbike.org/osm/bbbike/Beijing/Beijing.osm.pbf`)
- **Prepare:** run `geolife_audit.py` → `geolife_points.py` → `geolife_e0_semantic.py`
  (writes the point-level and enrichment parquets into `GEO_OUT`), then `geolife_e1..e4.py`.

Source: Zheng et al., *GeoLife: A Collaborative Social Networking Service among User,
Location and Trajectory*, IEEE Data Eng. Bull., 2010.

## `labeled_OSM_category.csv`, `pois_categories_OSM.txt`
Public OSM tag → semantic-label dictionaries used by `matsum_prepare.py`.
