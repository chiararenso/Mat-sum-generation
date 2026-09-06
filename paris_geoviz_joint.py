"""
Paris spatial layer: post-hoc DECOUPLED instantiation vs a JOINT spatial-semantic generator.
Decoupled: follow a pre-generated semantic Markov sequence and place each state's visit at a popular
tile (breaks locality -- the semantic sequence lost spatial adjacency).
Joint: generate directly over tiles. From the current tile, consider only tiles in a SPATIAL
NEIGHBOURHOOD, score each by  T[cur_state, tile_state] * popularity * exp(-d/scale)  and sample the next
tile. Semantics and geography are chosen together, so jumps stay local (as in reality).
Shows Real | Decoupled | Joint example trajectories + jump-length / radius-of-gyration distributions,
and reports the semantic fidelity (label bigram TV) the joint generator trades for locality.
"""
import os, numpy as np, pandas as pd, geopandas as gpd, warnings
from collections import Counter, defaultdict
import matsum_summarize as MS
from val_rq3 import rle, markov
warnings.filterwarnings("ignore")

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", ".")
K, M, G, CAP = 5, 581, 90, 300
DEC = dict(scale=1500.0, wpow=1.0, rho=0.6, gamma=0.21)
BS, NB, JSCALE = 250.0, 3, 300.0                      # joint: bucket size m, neighbourhood radius, decay
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def km(P):
    x = P[:, 0] * 111.320 * np.cos(np.radians(48.85)); y = P[:, 1] * 110.540; return x, y


def jumps_rg(trajs):
    J, RG = [], []
    for tr in trajs:
        P = np.asarray(tr)
        if len(P) < 2: continue
        x, y = km(P); d = np.hypot(np.diff(x), np.diff(y)); J.extend(d.tolist())
        RG.append(float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2))))
    return np.array(J), np.array(RG)


def bigram(seqs):
    c = Counter()
    for v in seqs:
        for a, b in zip(v, v[1:]): c[(a, b)] += 1
    n = sum(c.values()) or 1; return {k: x / n for k, x in c.items()}


def tv(a, b): return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))
def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0]))); return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    print("[joint] load + semantic mapping", flush=True)
    gdf, areas = MS.load(); sem = MS.semantic_mapping(gdf, areas)
    sem["lon"] = sem.geometry.x.values; sem["lat"] = sem.geometry.y.values
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem = sem.sort_values(["tid", "time"]).reset_index(drop=True)
    raw = {t: rle(list(g["sig2"])) for t, g in sem.groupby("tid", sort=False)}
    fs = Counter(s for t in raw for s in raw[t]); freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    fset = set(freq); remap = {s: (s if s in fset else max(freq, key=lambda f: jac(s, f))) for s in fs}
    sem["state"] = sem["sig2"].map(lambda s: remap.get(s, s))

    stc = sem["state"].values; tid = sem["tid"].values
    ch = np.empty(len(sem), bool); ch[0] = True; ch[1:] = (stc[1:] != stc[:-1]) | (tid[1:] != tid[:-1])
    sem["run"] = np.cumsum(ch)
    rv = sem.groupby("run").agg(tid=("tid", "first"), state=("state", "first"), top1=("top1", "first"),
                                lon=("lon", "mean"), lat=("lat", "mean")).reset_index(drop=True)
    real_tr = [g[["lon", "lat"]].to_numpy() for _, g in rv.groupby("tid", sort=False)]
    gt_bi = bigram([list(g["top1"]) for _, g in rv.groupby("tid", sort=False)])

    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    w = sem.groupby("index_right").size(); tiles["w"] = w.reindex(range(len(tiles))).fillna(0).values + 0.1
    tiles["sig2"] = tiles["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    tiles["state"] = tiles["sig2"].map(lambda s: remap.get(s, s))
    tiles["dom"] = tiles["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    cen = tiles.geometry.centroid; tiles["lon"] = cen.x.values; tiles["lat"] = cen.y.values
    xy = gpd.GeoSeries(cen, crs=tiles.crs).to_crs(2154); tiles["x"] = xy.x.values; tiles["y"] = xy.y.values

    states = sorted(set(sem.state.unique()), key=lambda z: sorted(z)); sidx = {s: i for i, s in enumerate(states)}; Sn = len(states)
    # transition matrix T[i,j] (row-normalised, Laplace)
    T = np.ones((Sn, Sn)) * 0.05
    for t in raw:
        seq = rle([remap[s] for s in raw[t]])
        for a, b in zip(seq, seq[1:]): T[sidx[a], sidx[b]] += 1
    T /= T.sum(1, keepdims=True)

    # tile arrays + spatial buckets; keep only tiles whose state is in vocab
    tv_ = tiles[tiles.state.isin(sidx)].reset_index(drop=True)
    TX, TY = tv_.x.to_numpy(), tv_.y.to_numpy(); TLON, TLAT = tv_.lon.to_numpy(), tv_.lat.to_numpy()
    TW = tv_.w.to_numpy(); TSI = tv_.state.map(sidx).to_numpy(); TDOM = tv_.dom.to_numpy()
    buckets = defaultdict(list)
    for i in range(len(tv_)): buckets[(int(TX[i] // BS), int(TY[i] // BS))].append(i)
    buckets = {k: np.array(v) for k, v in buckets.items()}

    # ---------- decoupled (post-hoc), capped popular tiles per state ----------
    STT = {}
    for s, g in tv_.groupby("state"):
        g = g.nlargest(min(CAP, len(g)), "w")
        STT[sidx[s]] = (g["x"].to_numpy(), g["y"].to_numpy(), g["lon"].to_numpy(), g["lat"].to_numpy(), g["w"].to_numpy())
    def dec_inst(seq_i, rng):
        visited = {}; pos = None; pts = []
        for zi in seq_i:
            c = STT.get(zi)
            if c is None: continue
            xs, ys, lons, lats, ws = c; nv = len(visited)
            p_exp = 1.0 if nv == 0 else min(1.0, DEC["rho"] * nv ** (-DEC["gamma"]))
            here = [k for k in visited if k[0] == zi]
            if here and rng.random() > p_exp:
                cc = np.array([visited[k] for k in here], float); k = here[int(rng.choice(len(here), p=cc / cc.sum()))]; j = k[1]
            else:
                if pos is None: p = ws / ws.sum()
                else:
                    p = ws * np.exp(-np.hypot(xs - pos[0], ys - pos[1]) / DEC["scale"]); ssum = p.sum(); p = p / ssum if ssum > 0 else ws / ws.sum()
                j = int(rng.choice(len(xs), p=p)); k = (zi, j); visited.setdefault(k, 0)
            visited[k] = visited.get(k, 0) + 1; pos = (xs[j], ys[j]); pts.append((lons[j], lats[j]))
        return pts

    # ---------- joint spatial-semantic generator ----------
    lengths = [len(t) for t in real_tr if len(t) >= 2]
    start_p = TW / TW.sum()
    def joint(rng):
        L = int(rng.choice(lengths)); cur = int(rng.choice(len(TX), p=start_p))
        pts = [(TLON[cur], TLAT[cur])]; sseq = [TSI[cur]]
        for _ in range(L - 1):
            bx, by = int(TX[cur] // BS), int(TY[cur] // BS)
            cand = np.concatenate([buckets[(bx + a, by + b)] for a in range(-NB, NB + 1) for b in range(-NB, NB + 1)
                                   if (bx + a, by + b) in buckets]) if True else np.array([])
            if len(cand) == 0: break
            d = np.hypot(TX[cand] - TX[cur], TY[cand] - TY[cur])
            score = T[TSI[cur], TSI[cand]] * TW[cand] * np.exp(-d / JSCALE)
            s = score.sum()
            nxt = cand[int(rng.choice(len(cand), p=score / s))] if s > 0 else cand[int(rng.choice(len(cand)))]
            cur = int(nxt); pts.append((TLON[cur], TLAT[cur])); sseq.append(TSI[cur])
        return pts, sseq

    real_seqs = [rle([remap[s] for s in raw[t]]) for t in raw]
    seqs = markov(real_seqs, M, np.random.default_rng(0))
    seqs_i = [[sidx[z] for z in s if z in sidx] for s in seqs]
    dtr = [dec_inst(s, np.random.default_rng(1)) for s in seqs_i]
    jrng = np.random.default_rng(2); jout = [joint(jrng) for _ in range(M)]
    jtr = [p for p, _ in jout]; jseq_lab = [[TDOM[0]]] * 0  # placeholder
    # joint semantic label sequences for fidelity: map joint state seq -> dominant label via tile dom already in pts? use state->dom
    state_dom = {}
    for s, g in sem.groupby("state"): state_dom[sidx[s]] = g["top1"].value_counts().index[0]
    jlab = [rle([state_dom.get(si, "NA") for si in ss]) for _, ss in jout]

    Jr, RGr = jumps_rg(real_tr); Jd, RGd = jumps_rg(dtr); Jj, RGj = jumps_rg(jtr)
    print(f"  median jump km: real={np.median(Jr):.2f} decoupled={np.median(Jd):.2f} joint={np.median(Jj):.2f}", flush=True)
    print(f"  median r_g km : real={np.median(RGr):.2f} decoupled={np.median(RGd):.2f} joint={np.median(RGj):.2f}", flush=True)
    print(f"  semantic label bigram-TV (vs real): joint={tv(bigram(jlab), gt_bi):.3f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    lo_lon, hi_lon = np.percentile(rv.lon, [0.5, 99.5]); lo_lat, hi_lat = np.percentile(rv.lat, [0.5, 99.5]); ext = [lo_lon, hi_lon, lo_lat, hi_lat]
    Rd = np.bincount((np.clip(((rv.lat - lo_lat) / (hi_lat - lo_lat) * G).astype(int), 0, G - 1) * G +
                      np.clip(((rv.lon - lo_lon) / (hi_lon - lo_lon) * G).astype(int), 0, G - 1)), minlength=G * G).astype(float)
    bg = np.where(Rd > 0, Rd / Rd.sum(), np.nan).reshape(G, G); vmax = np.nanmax(bg); vmin = vmax * 1e-3
    def pick(trs, n, seed):
        idx = [i for i, t in enumerate(trs) if 8 <= len(t) <= 22]; return [trs[i] for i in np.random.default_rng(seed).choice(idx, n, replace=False)]
    cols = plt.cm.tab10(np.arange(6))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.8))
    for a, trs, t in [(ax[0], pick(real_tr, 6, 6), "(A) Real"), (ax[1], pick(dtr, 6, 5), "(B) Synthetic — decoupled (post-hoc)"),
                      (ax[2], pick(jtr, 6, 7), "(C) Synthetic — joint spatial-semantic")]:
        a.imshow(bg, origin="lower", extent=ext, cmap="Greys", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto", alpha=.5)
        for c, tr in zip(cols, trs):
            P = np.asarray(tr); a.plot(P[:, 0], P[:, 1], "-o", color=c, lw=1.7, ms=3, alpha=.9); a.plot(P[0, 0], P[0, 1], "*", color=c, ms=14, mec="k", mew=.5)
        a.set_title(t); a.set_xlabel("lon"); a.set_ylabel("lat"); a.set_xlim(lo_lon, hi_lon); a.set_ylim(lo_lat, hi_lat)
    fig.suptitle("Paris · trajectories: real vs decoupled (post-hoc) vs joint spatial-semantic generation (★=start)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "fig_paris_joint_traj.png"), dpi=130); plt.close(fig); print("wrote fig_paris_joint_traj.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); bins = np.logspace(-1.6, 1.6, 42)
    for J, lab, c in [(Jr, f"real (med {np.median(Jr):.2f})", "#333"), (Jd, f"decoupled (med {np.median(Jd):.2f})", "#d95f0e"), (Jj, f"joint (med {np.median(Jj):.2f})", "#2c7fb8")]:
        ax[0].hist(J, bins=bins, density=True, histtype="step", lw=2.2, color=c, label=lab); ax[0].axvline(np.median(J), color=c, ls=":", lw=1.2)
    ax[0].set_xscale("log"); ax[0].set_xlabel("jump length between visits (km)"); ax[0].set_ylabel("density"); ax[0].set_title("(A) Jump-length distribution"); ax[0].legend(fontsize=9)
    for RGx, lab, c in [(RGr, f"real (med {np.median(RGr):.2f})", "#333"), (RGd, f"decoupled (med {np.median(RGd):.2f})", "#d95f0e"), (RGj, f"joint (med {np.median(RGj):.2f})", "#2c7fb8")]:
        ax[1].hist(RGx, bins=bins, density=True, histtype="step", lw=2.2, color=c, label=lab); ax[1].axvline(np.median(RGx), color=c, ls=":", lw=1.2)
    ax[1].set_xscale("log"); ax[1].set_xlabel("radius of gyration (km)"); ax[1].set_ylabel("density"); ax[1].set_title("(B) Radius of gyration"); ax[1].legend(fontsize=9)
    fig.suptitle("Paris · joint generation recovers the real short-jump / compact structure", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, "fig_paris_joint_stats.png"), dpi=130); plt.close(fig); print("wrote fig_paris_joint_stats.png")


if __name__ == "__main__":
    main()
