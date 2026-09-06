"""
Geographic comparison real vs synthetic on Paris -- approach (b): EXPLICIT spatial instantiation.
The MAT-Sum+Markov generator produces semantic-state sequences; we materialize each visit at a CONCRETE
tile using a decoupled EPR (exploration / preferential-return) placement:
  - explore with prob p = min(1, rho * S^-gamma) (S = #distinct visited tiles): pick a NEW tile of the
    current state, weighted by real popularity * spatial-proximity decay exp(-d/scale);
  - otherwise RETURN to an already-visited tile of that state, with prob proportional to past visits.
This yields an INDEPENDENT synthetic geography (the model chooses which tile), so the density comparison
and the example synthetic trajectories are a genuine test of spatial placement, not a reweighting.
Outputs: (1) real | synthetic(EPR) | log-ratio density; (2) example real vs synthetic trajectories.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
import matsum_summarize as MS
from val_rq3 import rle, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", ".")
K, M, G = 5, 581, 90
RHO, GAMMA, SCALE, CAP = 0.6, 0.21, 1500.0, 300
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    print("[geoviz-epr] load Paris + semantic mapping", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    sem["lon"] = sem.geometry.x.values; sem["lat"] = sem.geometry.y.values
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    sem = sem.sort_values(["tid", "time"]).reset_index(drop=True)

    raw = {t: rle(list(g["sig2"])) for t, g in sem.groupby("tid", sort=False)}
    fs = Counter(s for t in raw for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    fset = set(freq); remap = {s: (s if s in fset else max(freq, key=lambda f: jac(s, f))) for s in fs}
    sem["state"] = sem["sig2"].map(lambda s: remap.get(s, s))

    # ---- real visits (run-length over (tid,state)); location = mean point of the run ----
    st = sem["state"].values; tid = sem["tid"].values
    ch = np.empty(len(sem), bool); ch[0] = True; ch[1:] = (st[1:] != st[:-1]) | (tid[1:] != tid[:-1])
    sem["run"] = np.cumsum(ch)
    rv = sem.groupby("run").agg(tid=("tid", "first"), state=("state", "first"),
                                lon=("lon", "mean"), lat=("lat", "mean")).reset_index(drop=True)
    print(f"  points={len(sem)} states={sem.state.nunique()} real visits={len(rv)}", flush=True)

    # ---- tiles: state, centroid (lon/lat + metric x/y), popularity weight ----
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    w = sem.groupby("index_right").size()
    tiles["w"] = w.reindex(range(len(tiles))).fillna(0).values + 0.1
    tiles["sig2"] = tiles["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    tiles["state"] = tiles["sig2"].map(lambda s: remap.get(s, s))
    cen = tiles.geometry.centroid
    tiles["lon"] = cen.x.values; tiles["lat"] = cen.y.values
    xy = gpd.GeoSeries(cen, crs=tiles.crs).to_crs(2154)
    tiles["x"] = xy.x.values; tiles["y"] = xy.y.values

    state_tiles = {}
    for stt, g in tiles.groupby("state"):
        g = g.nlargest(min(CAP, len(g)), "w")
        state_tiles[stt] = (g["x"].to_numpy(), g["y"].to_numpy(), g["lon"].to_numpy(),
                            g["lat"].to_numpy(), g["w"].to_numpy())

    def instantiate(seq, rng):
        visited = {}                      # local tile-key -> count ; key = (state, j) index into candidate arr
        pos = None; pts = []
        for z in seq:
            c = state_tiles.get(z)
            if c is None:
                continue
            xs, ys, lons, lats, ws = c
            nvis = len(visited)
            p_exp = 1.0 if nvis == 0 else min(1.0, RHO * nvis ** (-GAMMA))
            here = [k for k in visited if k[0] == z]
            if here and rng.random() > p_exp:                     # preferential return
                cc = np.array([visited[k] for k in here], float)
                k = here[int(rng.choice(len(here), p=cc / cc.sum()))]
                j = k[1]
            else:                                                  # explore
                if pos is None:
                    p = ws / ws.sum()
                else:
                    dd = np.exp(-np.hypot(xs - pos[0], ys - pos[1]) / SCALE)
                    p = ws * dd; ssum = p.sum(); p = p / ssum if ssum > 0 else ws / ws.sum()
                j = int(rng.choice(len(xs), p=p)); k = (z, j)
                visited.setdefault(k, 0)
            visited[k] = visited.get(k, 0) + 1
            pos = (xs[j], ys[j]); pts.append((lons[j], lats[j]))
        return pts

    # ---- generate synthetic semantic sequences and instantiate ----
    real_seqs = [rle([remap[s] for s in raw[t]]) for t in raw]
    grng = np.random.default_rng(0)
    synth_seqs = markov(real_seqs, M, grng)
    rng = np.random.default_rng(1)
    synth_traj = [instantiate(s, rng) for s in synth_seqs]
    syn_pts = np.array([p for tr in synth_traj for p in tr])
    print(f"  synthetic visits placed={len(syn_pts)}", flush=True)

    # ---- density grids (visit level) ----
    lo_lon, hi_lon = np.percentile(rv.lon, [0.5, 99.5]); lo_lat, hi_lat = np.percentile(rv.lat, [0.5, 99.5])
    def hist(lon, lat):
        ix = np.clip(((np.asarray(lon) - lo_lon) / (hi_lon - lo_lon) * G).astype(int), 0, G - 1)
        iy = np.clip(((np.asarray(lat) - lo_lat) / (hi_lat - lo_lat) * G).astype(int), 0, G - 1)
        return np.bincount(iy * G + ix, minlength=G * G).astype(float)
    R = hist(rv.lon, rv.lat); Syn = hist(syn_pts[:, 0], syn_pts[:, 1])
    Rn, Sn = R / R.sum(), Syn / Syn.sum()
    print(f"  JSD={jsd(R, Syn):.4f}  Pearson r={np.corrcoef(R, Syn)[0,1]:.4f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    ext = [lo_lon, hi_lon, lo_lat, hi_lat]; grid = lambda v: np.where(v > 0, v, np.nan).reshape(G, G)
    vmax = max(Rn.max(), Sn.max()); vmin = vmax * 1e-3

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, v, t in [(ax[0], Rn, "(A) Real visit density"), (ax[1], Sn, "(B) Synthetic — EPR instantiated")]:
        im = a.imshow(grid(v), origin="lower", extent=ext, cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
        a.set_title(t); a.set_xlabel("lon"); a.set_ylabel("lat"); plt.colorbar(im, ax=a, shrink=.8, label="share of visits")
    lr = np.full(G * G, np.nan); ok = (R > 0) & (Syn > 0)
    lr[ok] = np.log2(Sn[ok] / Rn[ok]); lim = np.nanpercentile(np.abs(lr), 98)
    im = ax[2].imshow(lr.reshape(G, G), origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax[2].set_title("(C) log$_2$(synth / real) · red=over, blue=under"); ax[2].set_xlabel("lon"); ax[2].set_ylabel("lat")
    plt.colorbar(im, ax=ax[2], shrink=.8, label="log-ratio")
    fig.suptitle(f"Paris · real vs synthetic visit geography — EPR instantiation · JSD={jsd(R,Syn):.3f}, r={np.corrcoef(R,Syn)[0,1]:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "fig_paris_epr_density.png"), dpi=130); plt.close(fig)
    print("wrote fig_paris_epr_density.png")

    # ---- example trajectories (moderate length 8..22) ----
    bg = grid(Rn)
    real_by_tid = {t: g for t, g in rv.groupby("tid", sort=False)}
    real_ex = [g for g in real_by_tid.values() if 8 <= len(g) <= 22]
    syn_ex = [tr for tr in synth_traj if 8 <= len(tr) <= 22]
    er = [real_ex[i] for i in np.random.default_rng(3).choice(len(real_ex), 6, replace=False)]
    es = [syn_ex[i] for i in np.random.default_rng(4).choice(len(syn_ex), 6, replace=False)]
    cols = plt.cm.tab10(np.arange(6))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, exs, t in [(ax[0], er, "(A) Real example trajectories"), (ax[1], es, "(B) Synthetic (EPR) example trajectories")]:
        a.imshow(bg, origin="lower", extent=ext, cmap="Greys", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto", alpha=.55)
        for c, tr in zip(cols, exs):
            P = np.array(tr[["lon", "lat"]]) if hasattr(tr, "columns") else np.array(tr)
            a.plot(P[:, 0], P[:, 1], "-o", color=c, lw=1.6, ms=3, alpha=.9)
            a.plot(P[0, 0], P[0, 1], "*", color=c, ms=13, mec="k", mew=.5)
        a.set_title(t); a.set_xlabel("lon"); a.set_ylabel("lat"); a.set_xlim(lo_lon, hi_lon); a.set_ylim(lo_lat, hi_lat)
    fig.suptitle("Paris · example trajectories (★=start) over the real visit-density background", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "fig_paris_epr_traj.png"), dpi=130); plt.close(fig)
    print("wrote fig_paris_epr_traj.png")


if __name__ == "__main__":
    main()
