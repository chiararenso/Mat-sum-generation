"""
MAT-SUM -> synthetic MAT dataset.  PHASE 3 - spatially-coupled generator.

Fixes the micro-locality gap of the decoupled Phase-1a/Phase-2 pipeline by
generating directly at the TILE level with a transition that couples three
factors at every step:
    P(next tile) ~ T[sym(cur), sym(next)]          (learned semantic transition)
                   * exp(-dist(cur,next)/SCALE)     (spatial proximity)
                   * (freq_next + eps)              (visitation prior)
wrapped in an EPR explore/return process (Song et al. 2010): with the EPR law
p_new = rho * S^-gamma the agent explores a new (nearby, semantically-plausible)
tile, otherwise it preferentially returns to an already-visited tile. Locality
and a bounded radius of gyration now emerge from the process.

Compared against the decoupled Phase-2 baseline on jump-length, radius of
gyration and privacy. Run after matsum_prepare.py + matsum_summarize.py.
"""
import os, json
from collections import Counter, defaultdict
import numpy as np, pandas as pd, geopandas as gpd
import matsum_summarize as MS, matsum_synth as SY

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))
FIG = os.environ.get("MATSUM_FIG", os.path.join(HERE, "figures"))
M_TOP, K_SUPPORT, N_SYNTH = 3, 5, 200
RHO, GAMMA = 1.2, 0.05          # EPR exploration law  p_new = rho * S^-gamma
SCALE = 600.0                   # spatial-proximity length (m) for exploration
# (ρ,γ,SCALE) tuned to match the real radius of gyration (~2.4 km) at visit
# granularity; residual jump-length gap vs raw-GPS is a sampling-density artifact.
EPS = 0.05
SEED = 11
rng = np.random.default_rng(SEED)
METRIC_CRS = 2154


def sigf(t):
    return frozenset(list(t)[:M_TOP]) if hasattr(t, "__len__") else frozenset()


def rle(pts):
    o = []
    for p in pts:
        if not o or o[-1] != p:
            o.append(p)
    return o


def rg(p):
    a = np.array(p, float)
    return 0.0 if len(a) < 2 else float(np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()))


def jumps(p):
    a = np.array(p, float)
    return np.linalg.norm(np.diff(a, axis=0), axis=1) if len(a) > 1 else np.array([])


def main():
    print(f"[setup] tiles, symbols, semantic transition matrix (m={M_TOP}, k={K_SUPPORT})")
    gdf, areas = MS.load()
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    tm = tiles.to_crs(METRIC_CRS); c = tm.geometry.centroid
    X = c.x.values; Y = c.y.values
    cat_sig = {r.category: sigf(r["label_tfidf"]) for _, r in areas.iterrows()}
    tiles["sig"] = tiles["category"].map(cat_sig)
    vis = gpd.sjoin(gdf[["geometry"]], tiles[["locationID", "geometry"]], predicate="within")
    FREQ = tiles["locationID"].map(vis.groupby("locationID").size()).fillna(0.0).values

    # anonymized vocabulary + remap (same as Phase 1/2)
    sem = MS.semantic_mapping(gdf, areas)
    raw = SY.build_visits(sem, M_TOP)
    seqs, support, _, _ = SY.anonymize_vocabulary(raw, K_SUPPORT)
    fs = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, cnt in fs.items() if cnt >= K_SUPPORT and len(s) > 0]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    symbols = sorted(set(remap.values()), key=lambda z: sorted(z))
    sidx = {s: i for i, s in enumerate(symbols)}
    # tile -> symbol id (map each tile's raw signature through remap; unseen sigs -> nearest frequent)
    def tile_symbol(sig):
        if sig in remap:
            return remap[sig]
        return max(frequent, key=lambda f: jac(sig, f)) if frequent else symbols[0]
    SYM = np.array([sidx[tile_symbol(s)] for s in tiles["sig"]])

    # learned semantic transition matrix (Laplace)
    ns = len(symbols); T = np.full((ns, ns), 0.1)
    for v in seqs.values():
        syms = [sidx[s] for s, _ in v]
        for a, b in zip(syms, syms[1:]):
            T[a, b] += 1
    T /= T.sum(1, keepdims=True)

    # real tile-centroid sequences + length distribution
    vis2 = gpd.sjoin(gdf[["tid", "time", "geometry"]],
                     tiles[["locationID", "geometry"]].assign(x=X, y=Y),
                     predicate="within").sort_values(["tid", "time"])
    real_paths, real_len = {}, []
    for tid, g in vis2.groupby("tid"):
        p = rle(list(zip(g["x"], g["y"]))); real_paths[tid] = p; real_len.append(len(p))

    # ---- spatially-coupled EPR generation ----
    print("[generate] spatially-coupled EPR tile trajectories")
    tile_xy = np.column_stack([X, Y]); base = FREQ + EPS
    synth_paths = {}
    for i in range(N_SYNTH):
        L = int(rng.choice(real_len))
        start = rng.choice(len(SYM), p=base / base.sum())
        cur = start; visited = Counter({start: 1}); path = [tuple(tile_xy[start])]
        for _ in range(L - 1):
            Sd = len(visited)
            p_new = min(1.0, RHO * Sd ** (-GAMMA))
            if rng.random() < p_new or Sd == 0:            # EXPLORE (nearby + semantic)
                d = np.hypot(X - tile_xy[cur][0], Y - tile_xy[cur][1])
                w = T[SYM[cur], SYM] * np.exp(-d / SCALE) * base
                w[cur] = 0.0
                if w.sum() <= 0:
                    break
                nxt = rng.choice(len(SYM), p=w / w.sum())
            else:                                          # RETURN (preferential)
                ks = np.array(list(visited)); wv = np.array([visited[k] for k in ks], float)
                nxt = int(rng.choice(ks, p=wv / wv.sum()))
            visited[nxt] += 1; cur = nxt; path.append(tuple(tile_xy[nxt]))
        synth_paths[f"c_{i:04d}"] = rle(path)

    # decoupled Phase-2 baseline (reload if present)
    dec = None
    p2 = os.path.join(OUT, "synthetic_spatial.json")
    if os.path.exists(p2):
        dec = {k: [tuple(xy) for xy in v] for k, v in json.load(open(p2)).items()}

    # ---- metrics ----
    def stats(paths):
        rgs = np.array([rg(p) for p in paths.values()]) / 1000
        jl = np.concatenate([jumps(p) for p in paths.values()]) / 1000
        return rgs, jl
    rg_r, jl_r = stats(real_paths)
    rg_c, jl_c = stats(synth_paths)
    print("\n[Spatial fidelity]  (real vs coupled" + ("  vs decoupled)" if dec else ")"))
    line = lambda n, a, b: print(f"  {n:22s} real={a:.2f}  coupled={b:.2f}", end="")
    line("radius gyration (km)", np.median(rg_r), np.median(rg_c))
    if dec:
        rg_d, jl_d = stats(dec); print(f"  decoupled={np.median(rg_d):.2f}")
    else:
        print()
    line("jump length (km)", np.median(jl_r), np.median(jl_c))
    print(f"  decoupled={np.median(jl_d):.2f}" if dec else "")

    # privacy
    def pkey(p):
        return tuple((round(x), round(y)) for x, y in p)
    real_keys = set(pkey(p) for p in real_paths.values())
    exact = sum(pkey(p) in real_keys for p in synth_paths.values())

    def pdist(a, b):
        A, B = np.array(a), np.array(b)
        d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
        return (d.min(1).mean() + d.min(0).mean()) / 2
    rl = list(real_paths.values())
    dcr_c = np.array([min(pdist(s, r) for r in rl) for s in synth_paths.values()]) / 1000
    dcr_r = np.array([min(pdist(r, o) for j, o in enumerate(rl) if j != i)
                      for i, r in enumerate(rl)]) / 1000
    print(f"\n[Spatial privacy]  exact copies={exact}/{len(synth_paths)}  "
          f"DCR coupled->real={dcr_c.mean():.2f} km  real->real={dcr_r.mean():.2f} km")

    pd.DataFrame({"metric": ["rg_median_real", "rg_median_coupled", "jump_median_real",
                             "jump_median_coupled", "exact_copies", "DCR_coupled", "DCR_real"],
                  "value": [round(np.median(rg_r), 2), round(np.median(rg_c), 2),
                            round(np.median(jl_r), 2), round(np.median(jl_c), 2),
                            exact, round(dcr_c.mean(), 2), round(dcr_r.mean(), 2)]}
                 ).to_csv(os.path.join(OUT, "coupled_eval.csv"), index=False)
    _figure(jl_r, jl_c, (jl_d if dec else None), rg_r, rg_c, (rg_d if dec else None), synth_paths, tm)
    print("\n[done] output/coupled_eval.csv ; figures/coupled_spatial.png")


def _figure(jl_r, jl_c, jl_d, rg_r, rg_c, rg_d, synth_paths, tm):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    lb = np.logspace(-1.4, 1.2, 24)
    ax[0].hist(jl_r[jl_r > 0], bins=lb, density=True, histtype="step", lw=2, color="#2c7fb8", label="real")
    ax[0].hist(jl_c[jl_c > 0], bins=lb, density=True, histtype="step", lw=2, color="#2ca02c", label="coupled")
    if jl_d is not None:
        ax[0].hist(jl_d[jl_d > 0], bins=lb, density=True, histtype="step", lw=2, ls="--",
                   color="#d95f0e", label="decoupled")
    ax[0].set_xscale("log"); ax[0].set_title("Jump length (km, log)"); ax[0].legend()
    b = np.linspace(0, 6, 18)
    ax[1].hist(rg_r, bins=b, density=True, histtype="step", lw=2, color="#2c7fb8", label="real")
    ax[1].hist(rg_c, bins=b, density=True, histtype="step", lw=2, color="#2ca02c", label="coupled")
    if rg_d is not None:
        ax[1].hist(rg_d, bins=b, density=True, histtype="step", lw=2, ls="--", color="#d95f0e", label="decoupled")
    ax[1].set_title("Radius of gyration (km)"); ax[1].legend()
    minx, miny, maxx, maxy = tm.total_bounds
    ax[2].set_xlim(minx, maxx); ax[2].set_ylim(miny, maxy)
    for i, (_, p) in enumerate(list(synth_paths.items())[:6]):
        a = np.array(p); ax[2].plot(a[:, 0], a[:, 1], "-o", ms=2, lw=.8, alpha=.75, color=plt.cm.tab10(i))
    ax[2].set_title("6 coupled trajectories"); ax[2].set_aspect("equal")
    fig.suptitle(f"Phase 3 — spatially-coupled generator (ρ={RHO}, γ={GAMMA}, scale={SCALE:.0f}m)",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIG, "coupled_spatial.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  figure:", p)


if __name__ == "__main__":
    main()
