"""
Geographic comparison real vs synthetic on Paris (semantic-driven, no extra spatial model).
The generator produces semantic-state visits; we place them on the map via the state->space mapping:
each synthetic visit of state z is spread over the grid cells where z really occurs, weighted by the
real within-state occupancy P_real(cell | z). Then
    synth_density(cell) = sum_z  f_synth(z) * P_real(cell | z),
    real_density(cell)  = sum_z  f_real(z)  * P_real(cell | z)  (= the actual visit histogram).
The DIFFERENCE map therefore isolates where the generator's semantic-frequency error projects onto
geography. We report JSD, Pearson r and cosine over the two density fields, and a per-category panel.
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS
from val_rq3 import rle, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", ".")
K, N_GEN, G = 5, 2000, 90            # min-support, #synthetic sequences, grid resolution
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    print("[geoviz] load Paris + semantic mapping", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    sem["lon"] = sem.geometry.x.values; sem["lat"] = sem.geometry.y.values
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem = sem.sort_values(["tid", "time"]).reset_index(drop=True)

    # --- MAT-Sum vocabulary (m=2, k=5) on the full dataset ---
    raw = {t: rle(list(g["sig2"])) for t, g in sem.groupby("tid", sort=False)}
    fs = Counter(s for t in raw for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    fset = set(freq)
    remap = {s: (s if s in fset else max(freq, key=lambda f: jac(s, f))) for s in fs}
    sem["state"] = sem["sig2"].map(lambda s: remap.get(s, s))
    states = sorted(sem["state"].unique(), key=lambda z: sorted(z)); sidx = {s: i for i, s in enumerate(states)}
    S = len(states)
    dom = {st: sem.loc[sem.state == st, "top1"].value_counts().index[0] for st in states}
    print(f"  points={len(sem)}  states={S}", flush=True)

    # --- spatial grid (trim 0.5/99.5 pct to avoid outliers) ---
    lo_lon, hi_lon = np.percentile(sem.lon, [0.5, 99.5]); lo_lat, hi_lat = np.percentile(sem.lat, [0.5, 99.5])
    m = (sem.lon >= lo_lon) & (sem.lon <= hi_lon) & (sem.lat >= lo_lat) & (sem.lat <= hi_lat)
    d = sem[m]
    ix = np.clip(((d.lon - lo_lon) / (hi_lon - lo_lon) * G).astype(int), 0, G - 1)
    iy = np.clip(((d.lat - lo_lat) / (hi_lat - lo_lat) * G).astype(int), 0, G - 1)
    flat = (iy.values * G + ix.values)
    sarr = d["state"].map(sidx).values

    # H[state, cell] = real visit counts
    H = np.zeros((S, G * G)); np.add.at(H, (sarr, flat), 1.0)
    rowsum = H.sum(1); rowsum[rowsum == 0] = 1
    Pcell_given_state = H / rowsum[:, None]           # P_real(cell | state)
    f_real = H.sum(1) / H.sum()                        # real state frequency

    # --- synthetic state frequency from MAT-Sum + Markov generation ---
    real_seqs = [rle([remap[s] for s in raw[t]]) for t in raw]
    grng = np.random.default_rng(0)
    synth = markov(real_seqs, N_GEN, grng)
    cnt = Counter(s for v in synth for s in v)
    f_synth = np.array([cnt.get(states[i], 0) for i in range(S)], float); f_synth /= f_synth.sum()

    real_dens = (f_real[:, None] * Pcell_given_state).sum(0)
    synth_dens = (f_synth[:, None] * Pcell_given_state).sum(0)
    print(f"  JSD={jsd(real_dens, synth_dens):.4f}  Pearson r={np.corrcoef(real_dens, synth_dens)[0,1]:.4f}  "
          f"cosine={float(real_dens@synth_dens/(np.linalg.norm(real_dens)*np.linalg.norm(synth_dens))):.4f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    ext = [lo_lon, hi_lon, lo_lat, hi_lat]
    def grid(v): return np.where(v > 0, v, np.nan).reshape(G, G)
    R = real_dens / real_dens.sum(); Syn = synth_dens / synth_dens.sum()
    vmax = np.nanmax([R.max(), Syn.max()]); vmin = max(min(R[R > 0].min(), Syn[Syn > 0].min()), vmax * 1e-4)

    # ---------- Figure 1: Real | Synthetic | log-ratio ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, v, t in [(ax[0], R, "(A) Real visit density"), (ax[1], Syn, "(B) Synthetic (semantic-driven)")]:
        im = a.imshow(grid(v), origin="lower", extent=ext, cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
        a.set_title(t); a.set_xlabel("lon"); a.set_ylabel("lat"); plt.colorbar(im, ax=a, shrink=.8, label="share of visits")
    lr = np.full(G * G, np.nan); ok = (real_dens > 0) & (synth_dens > 0)
    lr[ok] = np.log2(Syn.reshape(-1)[ok] / R.reshape(-1)[ok])
    lim = np.nanpercentile(np.abs(lr), 98)
    im = ax[2].imshow(lr.reshape(G, G), origin="lower", extent=ext, cmap="RdBu_r",
                      norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax[2].set_title("(C) log$_2$(synth / real)  ·  red=over, blue=under"); ax[2].set_xlabel("lon"); ax[2].set_ylabel("lat")
    plt.colorbar(im, ax=ax[2], shrink=.8, label="log-ratio")
    fig.suptitle(f"Paris · real vs synthetic visit geography (semantic-driven) · "
                 f"JSD={jsd(real_dens,synth_dens):.3f}, r={np.corrcoef(real_dens,synth_dens)[0,1]:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "fig_paris_geoviz.png"), dpi=130); plt.close(fig)
    print("wrote fig_paris_geoviz.png")

    # ---------- Figure 2: per-category small multiples ----------
    catmass = defaultdict(float)
    for i, st in enumerate(states):
        catmass[dom[st]] += f_real[i]
    cats = [c for c, _ in sorted(catmass.items(), key=lambda kv: -kv[1])[:6]]
    fig, ax = plt.subplots(2, len(cats), figsize=(3.1 * len(cats), 6.4))
    for j, c in enumerate(cats):
        idx = [i for i, st in enumerate(states) if dom[st] == c]
        rc = (f_real[idx, None] * Pcell_given_state[idx]).sum(0)
        sc = (f_synth[idx, None] * Pcell_given_state[idx]).sum(0)
        vmx = max(rc.max(), sc.max()) or 1; vmn = max(vmx * 1e-3, 1e-9)
        for row, v, lab in [(0, rc, "real"), (1, sc, "synth")]:
            im = ax[row, j].imshow(grid(v), origin="lower", extent=ext, cmap="viridis",
                                   norm=LogNorm(vmin=vmn, vmax=vmx), aspect="auto")
            ax[row, j].set_xticks([]); ax[row, j].set_yticks([])
            if row == 0:
                ax[row, j].set_title(f"{c}\n(JSD {jsd(rc+1e-12, sc+1e-12):.2f})", fontsize=10)
            if j == 0:
                ax[row, j].set_ylabel(lab, fontsize=11)
    fig.suptitle("Paris · per-semantic-category visit geography · real (top) vs synthetic (bottom)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIG, "fig_paris_geoviz_categories.png"), dpi=130); plt.close(fig)
    print("wrote fig_paris_geoviz_categories.png")


if __name__ == "__main__":
    main()
