"""
Paris spatial instantiation: DECOUPLED vs COUPLED EPR, side by side.
Same synthetic semantic itineraries (MAT-Sum+Markov) are materialized two ways:
  - decoupled: exploration weight = popularity * exp(-d/1500) -> jumps to far popular tiles (sprawl);
  - coupled:   exploration weight = popularity^0.3 * exp(-d/450), stronger preferential-return
               -> distance dominates -> compact, local trajectories.
Shows (1) Real | Decoupled | Coupled example trajectories (identical itineraries in panels 2 & 3),
and (2) the jump-length distribution real vs decoupled vs coupled (coupled recovers real locality).
Also reports aggregate density JSD/r and median jump / radius of gyration for each.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter
import matsum_summarize as MS
from val_rq3 import rle, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", ".")
K, M, G, CAP = 5, 581, 90, 300
DEC = dict(scale=1500.0, wpow=1.0, rho=0.6, gamma=0.21)
COUP = dict(scale=450.0, wpow=0.3, rho=0.5, gamma=0.35)
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
R_EARTH = None


def km(lon, lat):
    """equirectangular meters->km helper: returns arrays of x_km,y_km around Paris."""
    lat0 = 48.85
    x = (np.asarray(lon)) * 111.320 * np.cos(np.radians(lat0)); y = (np.asarray(lat)) * 110.540
    return x, y


def jumps_rg(trajs):
    J, RG = [], []
    for tr in trajs:
        P = np.asarray(tr)
        if len(P) < 2:
            continue
        x, y = km(P[:, 0], P[:, 1])
        d = np.hypot(np.diff(x), np.diff(y)); J.extend(d.tolist())
        cx, cy = x.mean(), y.mean(); RG.append(float(np.sqrt(np.mean((x - cx) ** 2 + (y - cy) ** 2))))
    return np.array(J), np.array(RG)


def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    print("[coupled] load + semantic mapping", flush=True)
    gdf, areas = MS.load(); sem = MS.semantic_mapping(gdf, areas)
    sem["lon"] = sem.geometry.x.values; sem["lat"] = sem.geometry.y.values
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    sem = sem.sort_values(["tid", "time"]).reset_index(drop=True)
    raw = {t: rle(list(g["sig2"])) for t, g in sem.groupby("tid", sort=False)}
    fs = Counter(s for t in raw for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    fset = set(freq); remap = {s: (s if s in fset else max(freq, key=lambda f: jac(s, f))) for s in fs}
    sem["state"] = sem["sig2"].map(lambda s: remap.get(s, s))

    st = sem["state"].values; tid = sem["tid"].values
    ch = np.empty(len(sem), bool); ch[0] = True; ch[1:] = (st[1:] != st[:-1]) | (tid[1:] != tid[:-1])
    sem["run"] = np.cumsum(ch)
    rv = sem.groupby("run").agg(tid=("tid", "first"), state=("state", "first"),
                                lon=("lon", "mean"), lat=("lat", "mean")).reset_index(drop=True)

    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    w = sem.groupby("index_right").size()
    tiles["w"] = w.reindex(range(len(tiles))).fillna(0).values + 0.1
    tiles["sig2"] = tiles["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    tiles["state"] = tiles["sig2"].map(lambda s: remap.get(s, s))
    import warnings; warnings.filterwarnings("ignore")
    cen = tiles.geometry.centroid; tiles["lon"] = cen.x.values; tiles["lat"] = cen.y.values
    xy = gpd.GeoSeries(cen, crs=tiles.crs).to_crs(2154); tiles["x"] = xy.x.values; tiles["y"] = xy.y.values
    STT = {}
    for s, g in tiles.groupby("state"):
        g = g.nlargest(min(CAP, len(g)), "w")
        STT[s] = (g["x"].to_numpy(), g["y"].to_numpy(), g["lon"].to_numpy(), g["lat"].to_numpy(), g["w"].to_numpy())

    def inst(seq, rng, P):
        visited = {}; pos = None; pts = []
        for z in seq:
            c = STT.get(z)
            if c is None:
                continue
            xs, ys, lons, lats, ws = c
            nv = len(visited); p_exp = 1.0 if nv == 0 else min(1.0, P["rho"] * nv ** (-P["gamma"]))
            here = [k for k in visited if k[0] == z]
            if here and rng.random() > p_exp:
                cc = np.array([visited[k] for k in here], float); k = here[int(rng.choice(len(here), p=cc / cc.sum()))]; j = k[1]
            else:
                if pos is None:
                    p = ws / ws.sum()
                else:
                    p = (ws ** P["wpow"]) * np.exp(-np.hypot(xs - pos[0], ys - pos[1]) / P["scale"])
                    ss = p.sum(); p = p / ss if ss > 0 else ws / ws.sum()
                j = int(rng.choice(len(xs), p=p)); k = (z, j); visited.setdefault(k, 0)
            visited[k] = visited.get(k, 0) + 1; pos = (xs[j], ys[j]); pts.append((lons[j], lats[j]))
        return pts

    real_seqs = [rle([remap[s] for s in raw[t]]) for t in raw]
    seqs = markov(real_seqs, M, np.random.default_rng(0))
    dtr = [inst(s, np.random.default_rng(1), DEC) for s in seqs]
    ctr = [inst(s, np.random.default_rng(1), COUP) for s in seqs]
    real_tr = [g[["lon", "lat"]].to_numpy() for _, g in rv.groupby("tid", sort=False)]

    lo_lon, hi_lon = np.percentile(rv.lon, [0.5, 99.5]); lo_lat, hi_lat = np.percentile(rv.lat, [0.5, 99.5])
    def hist(pts):
        p = np.asarray(pts); ix = np.clip(((p[:, 0] - lo_lon) / (hi_lon - lo_lon) * G).astype(int), 0, G - 1)
        iy = np.clip(((p[:, 1] - lo_lat) / (hi_lat - lo_lat) * G).astype(int), 0, G - 1)
        return np.bincount(iy * G + ix, minlength=G * G).astype(float)
    Rd = hist(rv[["lon", "lat"]].to_numpy())
    Dd = hist([p for tr in dtr for p in tr]); Cd = hist([p for tr in ctr for p in tr])
    Jr, RGr = jumps_rg(real_tr); Jd, RGd = jumps_rg(dtr); Jc, RGc = jumps_rg(ctr)
    print(f"  density r: decoupled={np.corrcoef(Rd,Dd)[0,1]:.3f} (JSD {jsd(Rd,Dd):.3f}) | "
          f"coupled={np.corrcoef(Rd,Cd)[0,1]:.3f} (JSD {jsd(Rd,Cd):.3f})", flush=True)
    print(f"  median jump km: real={np.median(Jr):.2f} decoupled={np.median(Jd):.2f} coupled={np.median(Jc):.2f}", flush=True)
    print(f"  median r_g km : real={np.median(RGr):.2f} decoupled={np.median(RGd):.2f} coupled={np.median(RGc):.2f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    ext = [lo_lon, hi_lon, lo_lat, hi_lat]
    bg = np.where(Rd > 0, Rd / Rd.sum(), np.nan).reshape(G, G); vmax = np.nanmax(bg); vmin = vmax * 1e-3
    # pick example itineraries (same synth indices for dec & coup)
    si = [i for i in range(len(seqs)) if 8 <= len(dtr[i]) <= 22 and 8 <= len(ctr[i]) <= 22]
    ri = [i for i, t in enumerate(real_tr) if 8 <= len(t) <= 22]
    es = np.random.default_rng(5).choice(si, 6, replace=False); er = np.random.default_rng(6).choice(ri, 6, replace=False)
    cols = plt.cm.tab10(np.arange(6))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.8))
    panels = [("(A) Real", [real_tr[i] for i in er]), ("(B) Synthetic — decoupled EPR", [dtr[i] for i in es]),
              ("(C) Synthetic — coupled EPR", [ctr[i] for i in es])]
    for a, (t, trs) in zip(ax, panels):
        a.imshow(bg, origin="lower", extent=ext, cmap="Greys", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto", alpha=.5)
        for c, tr in zip(cols, trs):
            P = np.asarray(tr); a.plot(P[:, 0], P[:, 1], "-o", color=c, lw=1.7, ms=3, alpha=.9)
            a.plot(P[0, 0], P[0, 1], "*", color=c, ms=14, mec="k", mew=.5)
        a.set_title(t); a.set_xlabel("lon"); a.set_ylabel("lat"); a.set_xlim(lo_lon, hi_lon); a.set_ylim(lo_lat, hi_lat)
    fig.suptitle("Paris · example trajectories: real vs decoupled vs coupled EPR (same synthetic itineraries in B,C; ★=start)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "fig_paris_coupled_traj.png"), dpi=130); plt.close(fig)
    print("wrote fig_paris_coupled_traj.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.logspace(-1.3, 1.6, 40)
    for J, lab, c in [(Jr, f"real (med {np.median(Jr):.2f})", "#333"),
                      (Jd, f"decoupled (med {np.median(Jd):.2f})", "#d95f0e"),
                      (Jc, f"coupled (med {np.median(Jc):.2f})", "#2c7fb8")]:
        ax[0].hist(J, bins=bins, density=True, histtype="step", lw=2.2, color=c, label=lab)
        ax[0].axvline(np.median(J), color=c, ls=":", lw=1.2)
    ax[0].set_xscale("log"); ax[0].set_xlabel("jump length between visits (km)"); ax[0].set_ylabel("density")
    ax[0].set_title("(A) Jump-length distribution"); ax[0].legend(fontsize=9)
    binsg = np.logspace(-1.3, 1.6, 40)
    for RGx, lab, c in [(RGr, f"real (med {np.median(RGr):.2f})", "#333"),
                        (RGd, f"decoupled (med {np.median(RGd):.2f})", "#d95f0e"),
                        (RGc, f"coupled (med {np.median(RGc):.2f})", "#2c7fb8")]:
        ax[1].hist(RGx, bins=binsg, density=True, histtype="step", lw=2.2, color=c, label=lab)
        ax[1].axvline(np.median(RGx), color=c, ls=":", lw=1.2)
    ax[1].set_xscale("log"); ax[1].set_xlabel("radius of gyration (km)"); ax[1].set_ylabel("density")
    ax[1].set_title("(B) Radius of gyration"); ax[1].legend(fontsize=9)
    fig.suptitle("Paris · spatial locality: coupled EPR recovers the real short-jump / compact structure", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, "fig_paris_coupled_stats.png"), dpi=130); plt.close(fig)
    print("wrote fig_paris_coupled_stats.png")


if __name__ == "__main__":
    main()
