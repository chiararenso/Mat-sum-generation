"""
MAT-SUM applied to the Akkodis (Paris) GPS dataset.
STAGE 2 - Semantic trajectory summarization:
  (a) semantic mapping: point -> semantic location, temporal-weight (duration),
  (b) summarize1: merge same-context locations (identical aspect set),
  (c) summarize2: merge similar semantic locations (cosine >= tau),
  (d) metrics: summarization rate S_rate (paper Sec. 5).

Faithful port of trajectory_summarization/summarization.py, GPS branch.
"""
import os, ast
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))

TAU = 0.9   # cosine similarity threshold (paper default)


def load():
    gdf = gpd.read_parquet(os.path.join(OUT, "trajectories.parquet"))
    areas_unified = gpd.read_parquet(os.path.join(OUT, "semantic_locations.parquet"))
    return gdf, areas_unified


# =========================================================================
# (a) semantic mapping  (gps branch of summarization.semantic_enrichment)
# =========================================================================
def semantic_mapping(gdf, areas):
    # Join points to the simple square tiles (fast STRtree) rather than to the
    # dissolved semantic-location multipolygons (whose city-spanning bounding
    # boxes defeat the spatial index). A tile carries the same category/label/
    # label_tfidf as its semantic location, so the result is identical.
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    sem = gpd.sjoin(gdf, tiles[["category", "label", "label_tfidf", "geometry"]],
                    predicate="within")
    sem = sem[~sem.index.duplicated(keep="first")]        # one location per point
    sem = sem.sort_values(["tid", "time"])
    # temporal weight = time to the next point of the same trajectory
    sem["end_time"] = sem.groupby("tid")["time"].shift(-1)
    sem["duration"] = (sem["end_time"] - sem["time"]).dt.total_seconds()
    # last point of each trajectory has no successor -> weight 0 (as in RLE/gps setups)
    sem["duration"] = sem["duration"].fillna(0.0)
    sem["aspects"] = sem["label_tfidf"].apply(lambda x: set(x) if isinstance(x, (list, np.ndarray)) else set())
    return sem


# =========================================================================
# (b) summarize1 - collapse points sharing the exact same semantic location
# =========================================================================
def summarize1(sem):
    # All points of the same (tid, category) share the same aspect set (identical
    # semantic context), so 'first' equals the union — a fast vectorised aggregate.
    agg = sem.groupby(["tid", "category"], sort=False).agg(
        n_points=("category", "size"),
        duration=("duration", "sum"),
        aspects=("aspects", "first"),
        label=("label", "first"),
    ).reset_index()
    return agg


# =========================================================================
# (c) summarize2 - merge semantic locations with cosine similarity >= tau
# =========================================================================
def build_similarity(areas, tau):
    df = areas[["category", "label_tfidf"]].copy()
    df["doc"] = df["label_tfidf"].apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else "")
    vec = CountVectorizer(binary=True)
    X = vec.fit_transform(df["doc"])
    M = 1 - pairwise_distances(X.toarray(), metric="cosine")
    cats = df["category"].tolist()
    sim = pd.DataFrame(M, index=cats, columns=cats)
    similar = {c: set(sim.columns[sim.loc[c] > tau]) for c in cats}
    return sim, similar


def summarize2(s1, similar):
    """Greedy merge (per trajectory) of categories that are pairwise similar,
    following the repo's order-based absorption. Returns representative rows."""
    out = []
    for tid, g in s1.groupby("tid"):
        g = g.sort_values("category")
        reps = []            # list of dicts (representative semantic locations)
        for _, row in g.iterrows():
            placed = False
            for rep in reps:
                if row["category"] in similar.get(rep["category"], set()):
                    rep["duration"] += row["duration"]
                    rep["n_points"] += row["n_points"]
                    rep["members"].add(row["category"])
                    rep["aspects"] = rep["aspects"] & row["aspects"]   # intersection (repr. sem. loc.)
                    rep["aspects_union"] |= row["aspects"]
                    placed = True
                    break
            if not placed:
                reps.append({"tid": tid, "category": row["category"],
                             "members": {row["category"]},
                             "n_points": row["n_points"],
                             "duration": row["duration"],
                             "aspects": set(row["aspects"]),
                             "aspects_union": set(row["aspects"])})
        out.extend(reps)
    df = pd.DataFrame(out)
    return df


# =========================================================================
# (d) metrics
# =========================================================================
def summarization_rate(sem, s2):
    """S_rate = 1 - R(T_hat)/R(T), averaged over trajectories.
    R(T)   = number of mapped GPS points in the trajectory (each point = 1 location).
    R(That)= number of representative semantic locations after summarize2."""
    npoints = sem.groupby("tid").size().rename("n_points")
    nreps   = s2.groupby("tid").size().rename("n_reps")
    m = pd.concat([npoints, nreps], axis=1).dropna()
    m["S_rate"] = 1 - m["n_reps"] / m["n_points"]
    return m


def main():
    print(f"[load] trajectories + semantic locations  (tau={TAU})")
    gdf, areas = load()

    print("[a] semantic mapping (temporal weights)")
    sem = semantic_mapping(gdf, areas)
    print(f"    {len(sem)} points mapped to a semantic location "
          f"({len(sem)/len(gdf)*100:.1f}% of {len(gdf)})")

    print("[b] summarize1 (identical semantic context)")
    s1 = summarize1(sem)
    print(f"    {s1.groupby('tid').size().mean():.1f} distinct semantic locations / trajectory (avg)")

    print("[c] summarize2 (cosine similarity >= tau)")
    sim, similar = build_similarity(areas, TAU)
    s2 = summarize2(s1, similar)
    print(f"    {s2.groupby('tid').size().mean():.1f} representative semantic locations / trajectory (avg)")

    print("[d] summarization rate")
    m = summarization_rate(sem, s2)
    print(m.round(3).to_string())
    print(f"\n    >>> dataset S_rate = {m['S_rate'].mean():.4f}  "
          f"(paper Geolife GPS squares res17 ~ 0.90-0.98)")

    # persist (serialize set columns as sorted strings for parquet)
    sem_out = sem.drop(columns=["aspects"]).copy()
    sem_out["aspects"] = sem["aspects"].apply(lambda s: str(sorted(s)))
    sem_out.to_parquet(os.path.join(OUT, "semantic_trajectories.parquet"))
    for d in (s1, s2):
        for c in ("aspects", "aspects_union", "members"):
            if c in d.columns:
                d[c] = d[c].apply(lambda s: str(sorted(s)))
    s1.to_parquet(os.path.join(OUT, "summarized1.parquet"))
    s2.to_parquet(os.path.join(OUT, "summarized2.parquet"))
    m.to_csv(os.path.join(OUT, "summarization_rate.csv"))
    print("\n[done] stage 2 outputs written to", OUT)


if __name__ == "__main__":
    main()
