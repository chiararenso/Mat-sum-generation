"""GeoLife E0 -- semantic rows: OSM coverage, enriched cells, semantic locations, distinct labels,
semantic visits per user, and the effective user count U after the semantic-visit filters
(>=10 semantic visits/session, >=3 valid sessions/user). No cross-session transitions (tid=user__session)."""
import os, numpy as np, pandas as pd, geopandas as gpd

OUT = os.environ["MATSUM_OUT"]; GEO = os.environ["GEO_OUT"]
MIN_VISITS, MIN_SESS = 10, 3

gdf = gpd.read_parquet(os.path.join(OUT, "trajectories.parquet"))
tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
areas = gpd.read_parquet(os.path.join(OUT, "semantic_locations.parquet"))

sem = gpd.sjoin(gdf[["tid", "time", "geometry"]], tiles[["category", "label_tfidf", "geometry"]], predicate="within")
sem = sem[~sem.index.duplicated(keep="first")].sort_values(["tid", "time"])
coverage = len(sem) / len(gdf)
sem["user"] = sem["tid"].str.split("__").str[0]
sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")

def nvis(a):
    a = a.to_numpy()
    return int(1 + (a[1:] != a[:-1]).sum()) if len(a) else 0
vis_sess = sem.groupby("tid")["category"].agg(nvis)
sess_user = sem.groupby("tid")["user"].first()
vd = pd.DataFrame({"visits": vis_sess, "user": sess_user})
per_user_visits = vd.groupby("user")["visits"].sum()
valid = vd[vd.visits >= MIN_VISITS]
valid_per_user = valid.groupby("user").size()
U = int((valid_per_user >= MIN_SESS).sum())
distinct_labels = sem["top1"].nunique()

S = pd.read_parquet(os.path.join(GEO, "geolife_sessions.parquet"))
P = pd.read_parquet(os.path.join(GEO, "geolife_points.parquet"))

print("\n================  E0 -- GeoLife dataset audit  ================")
print(f"{'original users':42s}: 182")
print(f"{'users with data in study area (>=1 kept sess)':42s}: {P.user.nunique()}")
print(f"{'sessions kept (>=80% in bbox)':42s}: {P.tid.nunique():,}")
print(f"{'points kept (clipped to bbox)':42s}: {len(P):,}")
print(f"{'median sessions/user (kept)':42s}: {P.groupby('user').session.nunique().median():.0f}")
print(f"{'points with >=1 OSM label (coverage)':42s}: {coverage*100:.1f}%")
print(f"{'enriched cells':42s}: {tiles.shape[0]:,}")
print(f"{'semantic locations':42s}: {areas.shape[0]:,}")
print(f"{'distinct OSM labels':42s}: {distinct_labels}")
print(f"{'median semantic visits/user':42s}: {per_user_visits.median():.0f}")
print(f"{'valid sessions (>=%d sem.visits)' % MIN_VISITS:42s}: {len(valid):,}")
print(f"{'>>> effective users U (>=%d valid sess)' % MIN_SESS:42s}: {U}")
pd.DataFrame({"metric": ["users_area", "sessions_kept", "points_kept", "coverage_pct", "enriched_cells",
                         "semantic_locations", "distinct_labels", "median_sem_visits_user", "U"],
              "value": [P.user.nunique(), P.tid.nunique(), len(P), round(coverage*100, 1), tiles.shape[0],
                        areas.shape[0], distinct_labels, int(per_user_visits.median()), U]}
             ).to_csv(os.path.join(OUT, "geolife_E0.csv"), index=False)
print("\nsaved geolife_E0.csv")
