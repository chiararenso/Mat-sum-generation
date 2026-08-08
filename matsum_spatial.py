"""
MAT-SUM -> synthetic MAT dataset.  PHASE 2 - spatial instantiation.

Turn synthetic *semantic* sequences (symbols) into synthetic trajectories over
REAL Paris geography: each symbol is realized at a concrete enriched tile of that
semantic type, chosen with an EPR-style gravity rule (preferential return by real
visitation frequency, distance-decay from the previous location). This adds
spatial realism while never copying a real path (tiles are picked anew, city-wide).

Evaluates spatial fidelity (radius of gyration, jump-length, visitation Zipf) and
spatial privacy (exact-path copies, spatial distance-to-closest-record).
Run after matsum_prepare.py + matsum_summarize.py.
"""
import os, json
from collections import Counter, defaultdict
import numpy as np, pandas as pd, geopandas as gpd
import matsum_summarize as MS
import matsum_synth as SY

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output")); FIG = os.environ.get("MATSUM_FIG", os.path.join(HERE, "figures"))
M_TOP, K_SUPPORT, N_SYNTH = 3, 5, 200
SCALE = 150.0       # distance-decay length (m): p ~ w * exp(-d/SCALE). Note: below ~120m the
                    # synthetic jump-length hits a ~0.9km floor set by the sparsity of same-type
                    # tiles — a spatial-blind semantic generator cannot reach real micro-locality.
EPS  = 0.05         # small weight for never-visited tiles (EPR exploration)
SEED = 7
rng = np.random.default_rng(SEED)
METRIC_CRS = 2154   # RGF93 / Lambert-93 (metres, France)


def sig(tfidf, m):
    return frozenset(list(tfidf)[:m]) if hasattr(tfidf, "__len__") else frozenset()


def rle_centroids(points_xy):
    """collapse consecutive identical tiles -> ordered list of centroids."""
    out = []
    for xy in points_xy:
        if not out or out[-1] != xy:
            out.append(xy)
    return out


def radius_of_gyration(coords):
    a = np.array(coords, dtype=float)
    if len(a) < 2:
        return 0.0
    cm = a.mean(axis=0)
    return float(np.sqrt(((a - cm) ** 2).sum(axis=1).mean()))


def jumps(coords):
    a = np.array(coords, dtype=float)
    return np.linalg.norm(np.diff(a, axis=0), axis=1) if len(a) > 1 else np.array([])


def main():
    print(f"[load] tiles + trajectories  (m={M_TOP}, k={K_SUPPORT})")
    gdf, areas = MS.load()
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    tiles_m = tiles.to_crs(METRIC_CRS)
    cen = tiles_m.geometry.centroid
    tiles["x"], tiles["y"] = cen.x.values, cen.y.values

    # signature per tile (via its category's label_tfidf)
    cat_sig = {r.category: sig(r["label_tfidf"], M_TOP) for _, r in areas.iterrows()}
    tiles["sig"] = tiles["category"].map(cat_sig)

    # real visitation per tile (point-in-tile join) -> preferential-return weight
    vis = gpd.sjoin(gdf[["geometry"]], tiles[["locationID", "geometry"]], predicate="within")
    tile_freq = vis.groupby("locationID").size()
    tiles["freq"] = tiles["locationID"].map(tile_freq).fillna(0.0)

    # anonymized vocabulary remap (same as Phase 1) so symbols match the generator
    sem = MS.semantic_mapping(gdf, areas)
    raw = SY.build_visits(sem, M_TOP)
    seqs, support, _, _ = SY.anonymize_vocabulary(raw, K_SUPPORT)
    # rebuild the raw->anon remap by matching on frequent signatures
    freq_sig = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, c in freq_sig.items() if c >= K_SUPPORT and len(s) > 0]
    def jacc(a, b):
        u = len(a | b); return len(a & b) / u if u else 0.0
    remap = {}
    for s in freq_sig:
        remap[s] = s if s in frequent else max(frequent, key=lambda f: jacc(s, f))

    # candidate tiles per anonymized symbol
    cand = defaultdict(list)
    for _, t in tiles.iterrows():
        s = t["sig"]
        if s in remap:
            cand[remap[s]].append((t["x"], t["y"], t["freq"] + EPS))
    cand = {k: (np.array([c[0] for c in v]), np.array([c[1] for c in v]),
               np.array([c[2] for c in v])) for k, v in cand.items()}
    print(f"  candidate tiles per symbol: mean={np.mean([len(v[0]) for v in cand.values()]):.0f}")

    # ---- real spatial trajectories (tile-centroid sequences) ----
    vis2 = gpd.sjoin(gdf[["tid", "time", "geometry"]], tiles[["locationID", "x", "y", "geometry"]],
                     predicate="within").sort_values(["tid", "time"])
    real_paths = {}
    for tid, g in vis2.groupby("tid"):
        real_paths[tid] = rle_centroids(list(zip(g["x"], g["y"])))

    # ---- generate synthetic semantic sequences and instantiate ----
    print("[generate] semantic sequences -> spatial instantiation (EPR gravity)")
    SY.rng = np.random.default_rng(SEED)
    model = SY.SemanticMarkov(alpha=0.1).fit(seqs)
    synth_paths = {}
    for i in range(N_SYNTH):
        syms = SY.seq_symbols(model.sample(120))
        coords, prev = [], None
        for s in syms:
            if s not in cand:
                continue
            xs, ys, w = cand[s]
            if prev is None:
                p = w / w.sum()
            else:
                d = np.sqrt((xs - prev[0]) ** 2 + (ys - prev[1]) ** 2)
                g = w * np.exp(-d / SCALE)
                p = g / g.sum() if g.sum() > 0 else w / w.sum()
            j = rng.choice(len(xs), p=p)
            prev = (xs[j], ys[j]); coords.append(prev)
        if coords:
            synth_paths[f"synth_{i:04d}"] = coords
    print(f"  {len(synth_paths)} synthetic spatial trajectories")

    # ---- spatial fidelity ----
    rg_r = np.array([radius_of_gyration(p) for p in real_paths.values()]) / 1000
    rg_s = np.array([radius_of_gyration(p) for p in synth_paths.values()]) / 1000
    jl_r = np.concatenate([jumps(p) for p in real_paths.values()]) / 1000
    jl_s = np.concatenate([jumps(p) for p in synth_paths.values()]) / 1000
    print("\n[Spatial fidelity]")
    print(f"  radius of gyration (km): real median={np.median(rg_r):.2f}  synth={np.median(rg_s):.2f}")
    print(f"  jump length (km):        real median={np.median(jl_r):.2f}  synth={np.median(jl_s):.2f}")

    # ---- spatial privacy ----
    def path_key(p):
        return tuple((round(x), round(y)) for x, y in p)
    real_keys = set(path_key(p) for p in real_paths.values())
    exact = sum(path_key(p) in real_keys for p in synth_paths.values())
    # spatial DCR: min mean-nearest-centroid distance to any real path
    def path_dist(a, b):
        A, B = np.array(a), np.array(b)
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        return (d.min(1).mean() + d.min(0).mean()) / 2
    rl = list(real_paths.values())
    dcr_s = np.array([min(path_dist(s, r) for r in rl) for s in synth_paths.values()]) / 1000
    dcr_r = []
    for i, r in enumerate(rl):
        others = rl[:i] + rl[i+1:]
        dcr_r.append(min(path_dist(r, o) for o in others))
    dcr_r = np.array(dcr_r) / 1000
    print("\n[Spatial privacy]")
    print(f"  exact spatial-path copies: {exact}/{len(synth_paths)}")
    print(f"  spatial DCR (km): synth->real mean={dcr_s.mean():.2f}  real->real mean={dcr_r.mean():.2f}")

    # persist + figures
    pd.DataFrame({"metric": ["rg_median_real_km", "rg_median_synth_km", "jump_median_real_km",
                             "jump_median_synth_km", "exact_path_copies", "DCR_synth_km", "DCR_real_km"],
                  "value": [round(np.median(rg_r), 2), round(np.median(rg_s), 2),
                            round(np.median(jl_r), 2), round(np.median(jl_s), 2),
                            exact, round(dcr_s.mean(), 2), round(dcr_r.mean(), 2)]}
                 ).to_csv(os.path.join(OUT, "spatial_eval.csv"), index=False)
    with open(os.path.join(OUT, "synthetic_spatial.json"), "w") as f:
        json.dump({k: [[round(x, 1), round(y, 1)] for x, y in v] for k, v in synth_paths.items()}, f)
    _figures(rg_r, rg_s, jl_r, jl_s, dcr_s, dcr_r, synth_paths, real_paths, tiles_m)
    print("\n[done] output/spatial_eval.csv, synthetic_spatial.json; figures/")


def _figures(rg_r, rg_s, jl_r, jl_s, dcr_s, dcr_r, synth_paths, real_paths, tiles_m):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    b = np.linspace(0, max(rg_r.max(), rg_s.max()), 20)
    ax[0, 0].hist(rg_r, bins=b, density=True, alpha=.6, color="#2c7fb8", label="real")
    ax[0, 0].hist(rg_s, bins=b, density=True, alpha=.6, color="#d95f0e", label="synthetic")
    ax[0, 0].set_title("Radius of gyration (km)"); ax[0, 0].legend()
    jr = jl_r[jl_r > 0]; js = jl_s[jl_s > 0]
    lb = np.logspace(-1.3, 1.4, 22)
    ax[0, 1].hist(jr, bins=lb, density=True, alpha=.6, color="#2c7fb8", label="real")
    ax[0, 1].hist(js, bins=lb, density=True, alpha=.6, color="#d95f0e", label="synthetic")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_title("Jump length (km, log)"); ax[0, 1].legend()
    ax[1, 0].hist(dcr_s, bins=18, density=True, alpha=.6, color="#d95f0e", label="synth→real")
    ax[1, 0].hist(dcr_r, bins=18, density=True, alpha=.6, color="#2c7fb8", label="real→real")
    ax[1, 0].set_title("Spatial distance-to-closest-record (km)"); ax[1, 0].legend()
    # map: a few synthetic trajectories over the tile extent
    minx, miny, maxx, maxy = tiles_m.total_bounds
    ax[1, 1].set_xlim(minx, maxx); ax[1, 1].set_ylim(miny, maxy)
    for i, (_, p) in enumerate(list(synth_paths.items())[:6]):
        a = np.array(p)
        ax[1, 1].plot(a[:, 0], a[:, 1], "-o", ms=2, lw=.8, alpha=.7, color=plt.cm.tab10(i))
    ax[1, 1].set_title("6 synthetic trajectories (Lambert-93 m)"); ax[1, 1].set_aspect("equal")
    fig.suptitle(f"Phase 2 — spatial instantiation  (m={M_TOP}, k={K_SUPPORT}, scale={SCALE:.0f}m)", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIG, "spatial_synth.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  figure:", p)


if __name__ == "__main__":
    main()
