"""
MAT-SUM applied to the Akkodis (Paris) GPS dataset.
STAGE 3 - Visualization: one Folium map per trajectory showing
  - raw GPS points (blue),
  - the semantic locations the trajectory traverses (filled polygons, one colour
    per distinct semantic context),
  - which locations get merged by the summarization (dashed grouping),
and a dataset-level overview map of all semantic locations.
"""
import os, ast
import numpy as np
import pandas as pd
import geopandas as gpd
import folium

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))
MAPS = os.path.join(HERE, "maps")
os.makedirs(MAPS, exist_ok=True)

PARIS = [48.8566, 2.3522]

# tab20 palette (hex) - avoids a matplotlib dependency
_TAB20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
          "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
          "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
          "#17becf", "#9edae5"]


def color_for(i, n):
    return _TAB20[i % len(_TAB20)]


def load():
    gdf   = gpd.read_parquet(os.path.join(OUT, "trajectories.parquet"))
    areas = gpd.read_parquet(os.path.join(OUT, "semantic_locations.parquet"))
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet"))  # per-tile category
    sem   = gpd.read_parquet(os.path.join(OUT, "semantic_trajectories.parquet"))
    s1    = pd.read_parquet(os.path.join(OUT, "summarized1.parquet"))
    s2    = pd.read_parquet(os.path.join(OUT, "summarized2.parquet"))
    rate  = pd.read_csv(os.path.join(OUT, "summarization_rate.csv"), index_col=0)
    return gdf, areas, tiles, sem, s1, s2, rate


def overview_map(tiles):
    """All enriched tiles over the study area, coloured by semantic-context id."""
    center = [tiles.geometry.centroid.y.mean(), tiles.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")
    n = len(tiles)
    for i, (_, r) in enumerate(tiles.iterrows()):
        folium.GeoJson(
            r["geometry"],
            style_function=lambda _f, col=color_for(int(r["category"]), n): {
                "fillColor": col, "color": col, "weight": 0.2, "fillOpacity": 0.55},
            tooltip=f"cat {r['category']}: {str(r['label'])[:120]}",
        ).add_to(m)
    m.save(os.path.join(MAPS, "overview_semantic_locations.html"))
    print("  overview_semantic_locations.html")


def traj_map(tid, gdf, tiles, sem, s2, rate):
    tg  = gdf[gdf.tid == tid]
    reps = s2[s2.tid == tid]

    # tiles actually traversed by this trajectory (point-in-tile spatial join)
    visited = gpd.sjoin(tiles.reset_index()[["locationID", "category", "label", "geometry"]],
                        tg[["geometry"]], predicate="contains").drop_duplicates("locationID")
    cats = sorted(visited["category"].unique())
    cat_color = {c: color_for(i, len(cats)) for i, c in enumerate(cats)}

    center = [tg.lat.mean(), tg.lon.mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # traversed tiles, coloured by semantic context (same colour = same context)
    loc_layer = folium.FeatureGroup(name="Traversed cells (by semantic context)")
    for _, t in visited.iterrows():
        col = cat_color[t["category"]]
        folium.GeoJson(
            t["geometry"],
            style_function=lambda _f, col=col: {
                "fillColor": col, "color": col, "weight": 1, "fillOpacity": 0.5},
            tooltip=f"cat {t['category']}: {str(t['label'])[:150]}",
        ).add_to(loc_layer)
    loc_layer.add_to(m)

    # raw GPS points + path
    raw = folium.FeatureGroup(name="Raw GPS points")
    folium.PolyLine(tg[["lat", "lon"]].values.tolist(),
                    color="#333", weight=2, opacity=0.6).add_to(raw)
    for _, p in tg.iterrows():
        folium.CircleMarker([p.lat, p.lon], radius=2, color="#111",
                            fill=True, fill_opacity=0.85).add_to(raw)
    raw.add_to(m)

    folium.LayerControl().add_to(m)
    r = rate.loc[tid] if tid in rate.index else None
    title = (f"<b>{tid[:8]}</b> — {len(tg)} pts, "
             f"{len(cats)} distinct cells → {len(reps)} repr. sem.loc.  "
             f"S_rate={r['S_rate']:.3f}" if r is not None else tid[:8])
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:8px;left:50px;z-index:9999;'
        f'background:white;padding:6px 10px;border-radius:6px;'
        f'font-family:sans-serif;font-size:13px;box-shadow:0 1px 4px #0003">{title}</div>'))
    m.save(os.path.join(MAPS, f"traj_{tid[:8]}.html"))


def main():
    gdf, areas, tiles, sem, s1, s2, rate = load()
    print("[overview]"); overview_map(tiles)
    print("[per-trajectory maps]")
    for tid in sorted(gdf.tid.unique()):
        traj_map(tid, gdf, tiles, sem, s2, rate)
    print(f"  {gdf.tid.nunique()} trajectory maps written to {MAPS}")


if __name__ == "__main__":
    main()
