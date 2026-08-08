"""
MAT-SUM applied to the Akkodis (Paris) GPS dataset.
STAGE 1 - Map preparation:
  (a) load trajectories, (b) tessellate the area (quadkey squares, res 17),
  (c) download + label OSM semantic aspects for Paris,
  (d) enrich tiles and dissolve into semantic locations.

Faithful re-implementation of chiarap2/MAT-Sum (SIGSPATIAL'23), adapted to the
Akkodis schema and cleaned of hard-coded paths / broken config plumbing.
tesspy is replaced by an equivalent mercantile quadkey grid (what tesspy.squares does).
"""
import os, re, json, ast, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import geopandas as gpd
import mercantile
from pyrosm import OSM
from shapely.geometry import Polygon, box
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
AKK  = os.path.join(os.path.dirname(HERE), "data (AKKODIS)")
# input trajectories + output dir are env-configurable so the same pipeline
# runs on other datasets (e.g. the NY+Paris MAT dataset). Default = Akkodis.
INPUT = os.environ.get("MATSUM_INPUT", os.path.join(AKK, "points_data.csv"))
OUT   = os.environ.get("MATSUM_OUT",   os.path.join(HERE, "output"))
os.makedirs(OUT, exist_ok=True)

POINTS_CSV = INPUT   # backward-compatible alias
TAGS_TXT   = os.path.join(DATA, "pois_categories_OSM.txt")
LABELS_CSV = os.path.join(DATA, "labeled_OSM_category.csv")
PBF_PATH   = os.environ.get("MATSUM_PBF", "")  # set via env; large file kept outside OneDrive

# ----- configuration (paper default: squares, resolution 17) -----
CONFIG = {
    "tessellation": {"type": "squares", "resolution": 17},
    "semantic_aspects": {"POIs": True, "landuse": True, "public_transport": True},
    "bbox_pad_m": 300,   # small padding so tiles fully cover boundary points
}

# =========================================================================
# (a) load trajectories
# =========================================================================
def load_trajectories():
    df = pd.read_parquet(INPUT) if INPUT.endswith(".parquet") else pd.read_csv(INPUT)
    ren = {"trajectory_id": "tid", "traj_id": "tid", "latitude": "lat",
           "longitude": "lon", "timestamp": "time"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["tid", "time"]).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(df[["tid", "lat", "lon", "time"]],
                           geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    return gdf


# =========================================================================
# (b) tessellation - quadkey squares at a given zoom (== tesspy.squares)
# =========================================================================
def tessellate(gdf, cfg):
    pad = cfg["bbox_pad_m"] / 111_000.0
    west, south = gdf.lon.min() - pad, gdf.lat.min() - pad
    east, north = gdf.lon.max() + pad, gdf.lat.max() + pad
    z = cfg["tessellation"]["resolution"]
    rows = []
    for t in mercantile.tiles(west, south, east, north, zooms=z):
        b = mercantile.bounds(t)
        rows.append({"locationID": mercantile.quadkey(t),
                     "geometry": box(b.west, b.south, b.east, b.north)})
    tiles = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    print(f"  tessellation: {len(tiles)} square tiles (zoom {z})")
    return tiles


# =========================================================================
# (c) OSM semantic aspects for the area
# =========================================================================
def read_categories(path):
    """Parse pois_categories_OSM.txt -> {osm_key: [values]}."""
    tags = {}
    with open(path) as f:
        tag = ""
        for line in f:
            if "=" not in line:
                continue
            key = re.findall(r".*=", line)[0][:-1]
            val = re.findall(r"=.*;", line)[0][1:-1]
            if tag != key:
                tag = key
                tags[tag] = []
            tags[tag].append(val)
    return tags


def extract_osm_features(bbox, tags):
    """Extract OSM features locally from the Ile-de-France PBF with pyrosm,
    clipped to the trajectories' bounding box, filtered to our tag dictionary.
    Replaces the (unreliable) live Overpass download with an offline extract."""
    cache = os.path.join(OUT, "osm_features_raw.parquet")
    if os.path.exists(cache):
        print("  OSM features: loading cached", cache)
        return gpd.read_parquet(cache)
    if not PBF_PATH or not os.path.exists(PBF_PATH):
        raise FileNotFoundError(
            f"PBF not found. Set MATSUM_PBF to the Ile-de-France .osm.pbf path (got: {PBF_PATH!r})")
    print(f"  OSM features: extracting from PBF (bbox-clipped) ...")
    osm = OSM(PBF_PATH, bounding_box=list(bbox))
    feats = osm.get_data_by_custom_criteria(
        custom_filter={k: v for k, v in tags.items()},
        filter_type="keep",
        keep_nodes=True, keep_ways=True, keep_relations=True,
        tags_as_columns=list(tags.keys()),
    )
    feats = feats.reset_index(drop=True)
    keep = [c for c in tags.keys() if c in feats.columns] + ["geometry"]
    feats = gpd.GeoDataFrame(feats[keep], geometry="geometry", crs="EPSG:4326")
    feats.to_parquet(cache)
    print(f"  OSM features: {len(feats)} geometries extracted")
    return feats


def melt_pois(feats, tags):
    """Replicates download_poi_osm: long form with OSM category / POI category."""
    macro = [c for c in tags.keys() if c in feats.columns]
    # pyrosm returns StringDtype columns; pandas-2.x stack() keeps NA rows for
    # that dtype, so drop them explicitly to get only real (geometry, tag) pairs.
    stacked = feats[macro].stack().dropna()              # (row, osm_key) -> value
    poi = stacked.reset_index()
    poi.columns = ["row", "OSM category", "POI category"]
    poi = poi.set_index("row")
    poi["geometry"] = feats.loc[poi.index, "geometry"].values
    poi["OSM category"] = poi["OSM category"].str.replace("_", " ")
    poi["POI category"] = poi["POI category"].astype(str).str.replace("_", " ")
    poi = poi[~poi["POI category"].str.contains("yes")]
    poi = poi[~poi["POI category"].str.contains(";")]
    poi = gpd.GeoDataFrame(poi, geometry="geometry", crs=feats.crs)
    return poi


def label_aspects(poi):
    """Map POI category -> semantic label via labeled_OSM_category.csv,
    then split into POIs / landuse / public_transport (as in labeling_aspects)."""
    labels = pd.read_csv(LABELS_CSV, delimiter=";")
    labels["poi_category"] = labels["poi_category"].str.replace("_", " ")
    lut = dict(zip(labels.poi_category, labels.label))
    poi["label"] = poi["POI category"].map(lut)
    n_unmapped = poi["label"].isna().sum()
    if n_unmapped:
        unk = sorted(str(x) for x in poi.loc[poi["label"].isna(), "POI category"].unique())
        print(f"  labeling: {n_unmapped} rows unmapped over {len(unk)} categories -> dropped")
        print("            e.g.", unk[:10])
    poi = poi[~poi["label"].isna()].copy()

    landuse = poi[poi["label"] == "landuse"].copy()
    public_transport = poi[poi["label"] == "public transport"].copy()
    pois = poi[~poi["label"].isin(["landuse", "public transport"])].copy()
    print(f"  labeling: POIs={len(pois)}  landuse={len(landuse)}  public_transport={len(public_transport)}")
    return pois, landuse, public_transport


# =========================================================================
# (d) enrich tiles and dissolve into semantic locations
# =========================================================================
def enrich_areas(locations, pois, landuse, public_transport):
    """Faithful port of enrichment.enrich_areas."""
    parts = []
    if len(pois):
        p = gpd.sjoin(pois, locations, predicate="intersects").set_index("locationID")
        parts.append(p)                                   # POIs keep their semantic label
    if len(landuse):
        l = gpd.sjoin(landuse, locations, predicate="intersects").set_index("locationID")
        l["label"] = l["POI category"]                    # landuse uses fine OSM value
        parts.append(l)
    if len(public_transport):
        t = gpd.sjoin(public_transport, locations, predicate="intersects").set_index("locationID")
        t["label"] = t["POI category"]                    # transport uses fine OSM value
        parts.append(t)

    locations = locations.set_index("locationID")
    sem = pd.concat(parts)

    areas = pd.DataFrame(sem.groupby("locationID")["label"]
                         .apply(lambda x: str(sorted(set(x)))))
    mapping = {v: i for i, v in enumerate(areas["label"].unique())}
    areas["category"] = areas["label"].map(mapping)
    areas["geometry"] = locations.loc[areas.index, "geometry"]
    areas = gpd.GeoDataFrame(areas, geometry="geometry", crs=locations.crs)

    # TF-IDF representation of each area's label document
    docs = sem[["label"]].copy()
    docs["label"] = docs["label"].str.replace(",", "").str.replace(" ", "_")
    docs = pd.DataFrame(docs.groupby("locationID")["label"].apply(lambda x: " ".join(x)))
    vec = TfidfVectorizer()
    m = vec.fit_transform(docs["label"].tolist())
    tfidf = pd.DataFrame(m.todense(), columns=vec.get_feature_names_out(), index=docs.index)
    tfidf_stack = tfidf.stack()
    tfidf_stack = tfidf_stack[tfidf_stack != 0].reset_index()
    tfidf_stack.columns = ["locationID", "term", "score"]
    tfidf_stack = tfidf_stack.sort_values("score", ascending=False)
    label_tfidf = tfidf_stack.groupby("locationID")["term"].apply(list)
    areas["label_tfidf"] = areas.index.map(label_tfidf)
    areas["label_tfidf"] = areas["label_tfidf"].apply(lambda x: x if isinstance(x, list) else [])

    # dissolve tiles that share identical semantic context -> semantic locations
    areas_unified = areas.dissolve(by="category", aggfunc="first",
                                   as_index=False, sort=True)
    print(f"  enrichment: {len(areas)} enriched tiles -> {len(areas_unified)} semantic locations")
    return areas, areas_unified


# =========================================================================
def main():
    print("[1] load trajectories"); gdf = load_trajectories()
    print(f"    {gdf.tid.nunique()} trajectories, {len(gdf)} points")

    print("[2] tessellation"); tiles = tessellate(gdf, CONFIG)
    bbox = tuple(tiles.total_bounds)  # (west, south, east, north)

    print("[3] OSM semantic aspects")
    tags = read_categories(TAGS_TXT)
    feats = extract_osm_features(bbox, tags)
    poi_long = melt_pois(feats, tags)
    pois, landuse, public_transport = label_aspects(poi_long)

    print("[4] enrich tiles -> semantic locations")
    areas, areas_unified = enrich_areas(tiles, pois, landuse, public_transport)

    gdf.to_parquet(os.path.join(OUT, "trajectories.parquet"))
    tiles.to_parquet(os.path.join(OUT, "tiles.parquet"))
    areas.to_parquet(os.path.join(OUT, "enriched_tiles.parquet"))
    areas_unified.to_parquet(os.path.join(OUT, "semantic_locations.parquet"))
    print("[done] stage 1 outputs written to", OUT)


if __name__ == "__main__":
    main()
