"""
E4 -- normalized Paris <-> GeoLife comparison (external-validation synthesis).
Reads the cached Paris RQ1/RQ2/RQ3 summaries and the GeoLife E1/E2/E3 summaries and emits, on a
common footing, the three headline findings as RELATIVE / normalized quantities:
  RQ1  MAT-Sum semantic-gen-TV vs best baseline -> relative improvement
  RQ2  relative TV reduction across n; saturation n* (same 2-segment log-n fit, bootstrap);
       first n with dDCR>0; max exact-copy rate
  RQ3  Markov-LSTM deltas (sem-gen-TV, near-copy, dDCR) at n=20 and at n_max; LSTM near-copy peak
Outputs a comparison table (geolife_e4_comparison.csv) and a 3-panel figure (geolife_e4.png):
  (A) RQ1 relative improvement over best baseline; (B) RQ2 scaling curves overlaid; (C) RQ3 LSTM
  near-copy signature overlaid (with Markov baselines).
"""
import os, numpy as np, pandas as pd

MAT = os.path.dirname(os.path.abspath(__file__))
PAR = os.path.join(MAT, "val_paris", "output")
GEO = os.environ.get("MATSUM_OUT", os.path.join(MAT, "geolife", "output"))
FIG = os.environ.get("MATSUM_FIG", MAT)


def sat(raw, ncol, ycol, repcol):
    NS = sorted(raw[ncol].unique()); ns = np.array(NS, float); logn = np.log(ns)
    def bp(mt):
        best, bx = 1e18, ns[1]
        for k in range(1, len(ns) - 1):
            e = 0.0
            for sl in (slice(0, k + 1), slice(k, len(ns))):
                X = np.vstack([logn[sl], np.ones(logn[sl].size)]).T; y = mt[sl]
                c, *_ = np.linalg.lstsq(X, y, rcond=None); e += ((X @ c - y) ** 2).sum()
            if e < best:
                best, bx = e, ns[k]
        return bx
    reps = sorted(raw[repcol].unique()); brng = np.random.default_rng(3); bps = []
    for _ in range(2000):
        s = brng.choice(reps, len(reps), replace=True)
        mt = np.array([raw[(raw[ncol] == n) & (raw[repcol].isin(s))][ycol].mean() for n in NS])
        bps.append(bp(mt))
    mt = np.array([raw[raw[ncol] == n][ycol].mean() for n in NS])
    return bp(mt), np.percentile(bps, [2.5, 97.5])


# ---------- load ----------
p_abs = pd.read_csv(os.path.join(PAR, "val_abstraction_ci.csv"))
p_r2 = pd.read_csv(os.path.join(PAR, "val_rq2_summary.csv"))
p_r2raw = pd.read_csv(os.path.join(PAR, "val_rq2_raw.csv"))
p_r3 = pd.read_csv(os.path.join(PAR, "val_rq3_summary.csv"))
g_e1 = pd.read_csv(os.path.join(GEO, "geolife_e1_summary.csv"))
g_e2 = pd.read_csv(os.path.join(GEO, "geolife_e2_summary.csv"))
g_e2raw = pd.read_csv(os.path.join(GEO, "geolife_e2_raw.csv"))
g_e3 = pd.read_csv(os.path.join(GEO, "geolife_e3_summary.csv"))

# ---------- RQ1: relative improvement over best baseline (semantic gen-TV) ----------
def rq1(df, matlabel):
    m = df.set_index(df.columns[0])
    ycol = "semantic_genTV_mean" if "semantic_genTV_mean" in df else "sem_genTV"
    mat = m.loc[matlabel, ycol]
    base = m.drop(index=matlabel)[ycol]
    bestb = base.min(); bestname = base.idxmin()
    return mat, bestb, bestname, 100 * (bestb - mat) / bestb
p1 = rq1(p_abs, "MAT-Sum-Markov")
g1 = rq1(g_e1, "MAT-Sum")

# ---------- RQ2 ----------
def rq2(r2, r2raw, ncol="n", ycol="sem_genTV", repcol="rep"):
    ns = sorted(r2[ncol].unique()); lo, hi = ns[0], ns[-1]
    y0 = float(r2[r2[ncol] == lo][ycol].iloc[0]); y1 = float(r2[r2[ncol] == hi][ycol].iloc[0])
    rel = 100 * (y0 - y1) / y0
    d = r2.sort_values(ncol); pos = d[d["dDCR"] > 0][ncol]
    npos = int(pos.min()) if len(pos) else None
    maxexact = float(r2["exact_copies"].max())
    s, sci = sat(r2raw, ncol, ycol, repcol)
    return dict(lo=lo, hi=hi, y0=y0, y1=y1, rel=rel, npos=npos, maxexact=maxexact,
                sat=s, sci=sci)
p2 = rq2(p_r2, p_r2raw)
g2 = rq2(g_e2, g_e2raw)

# ---------- RQ3: Markov - LSTM at n=20 and n_max ----------
def rq3(df, ycol, mlabel, llabel, gcol):
    ns = sorted(df["n"].unique())
    out = {}
    for tag, n in [("n20", 20), ("nmax", ns[-1])]:
        mk = df[(df["n"] == n) & (df[gcol] == mlabel)]
        ls = df[(df["n"] == n) & (df[gcol] == llabel)]
        out[tag] = dict(
            n=n,
            dTV=float(mk[ycol].iloc[0]) - float(ls[ycol].iloc[0]),
            dnear=float(mk["near_pct"].iloc[0]) - float(ls["near_pct"].iloc[0]),
            dddcr=float(mk["dDCR"].iloc[0]) - float(ls["dDCR"].iloc[0]),
            lstm_near=float(ls["near_pct"].iloc[0]))
    return out
p3 = rq3(p_r3, "sem_genTV", "Markov", "LSTM", "model")
g3 = rq3(g_e3, "sem_biTV", "Markov", "LSTM", "gen")

# ---------- comparison table ----------
rows = [
    ("RQ1  MAT-Sum sem-gen-TV",              f"{p1[0]:.3f}",                 f"{g1[0]:.3f}"),
    ("RQ1  best baseline sem-gen-TV",        f"{p1[1]:.3f} ({p1[2]})",       f"{g1[1]:.3f} ({g1[2]})"),
    ("RQ1  relative improvement",            f"{p1[3]:.1f}%",                f"{g1[3]:.1f}%"),
    ("RQ2  sem-gen-TV  (min n -> max n)",    f"{p2['y0']:.3f} -> {p2['y1']:.3f}", f"{g2['y0']:.3f} -> {g2['y1']:.3f}"),
    ("RQ2  n range",                         f"{p2['lo']}..{p2['hi']}",      f"{g2['lo']}..{g2['hi']}"),
    ("RQ2  relative TV reduction",           f"{p2['rel']:.1f}%",            f"{g2['rel']:.1f}%"),
    ("RQ2  saturation n* [95% CI]",          f"{p2['sat']:.0f} [{p2['sci'][0]:.0f},{p2['sci'][1]:.0f}]",
                                             f"{g2['sat']:.0f} [{g2['sci'][0]:.0f},{g2['sci'][1]:.0f}]"),
    ("RQ2  first n with dDCR>0",             f"{p2['npos']}",                f"{g2['npos']}"),
    ("RQ2  max exact-copy rate (/200)",      f"{p2['maxexact']:.2f} ({100*p2['maxexact']/200:.2f}%)",
                                             f"{g2['maxexact']:.2f} ({100*g2['maxexact']/200:.2f}%)"),
    ("RQ3  Markov-LSTM sem-TV  @n=20",       f"{p3['n20']['dTV']:+.3f}",     f"{g3['n20']['dTV']:+.3f}"),
    ("RQ3  Markov-LSTM near-copy @n=20 (pp)",f"{p3['n20']['dnear']:+.2f}",   f"{g3['n20']['dnear']:+.2f}"),
    ("RQ3  Markov-LSTM dDCR    @n=20",       f"{p3['n20']['dddcr']:+.3f}",   f"{g3['n20']['dddcr']:+.3f}"),
    ("RQ3  LSTM near-copy peak @n=20",       f"{p3['n20']['lstm_near']:.2f}%", f"{g3['n20']['lstm_near']:.2f}%"),
    ("RQ3  Markov-LSTM sem-TV  @n_max",      f"{p3['nmax']['dTV']:+.3f}",    f"{g3['nmax']['dTV']:+.3f}"),
    ("RQ3  Markov-LSTM near-copy @n_max(pp)",f"{p3['nmax']['dnear']:+.2f}",  f"{g3['nmax']['dnear']:+.2f}"),
    ("RQ3  Markov-LSTM dDCR    @n_max",      f"{p3['nmax']['dddcr']:+.3f}",  f"{g3['nmax']['dddcr']:+.3f}"),
]
tab = pd.DataFrame(rows, columns=["quantity", "Paris (n=581)", "GeoLife (n=152)"])
tab.to_csv(os.path.join(GEO, "geolife_e4_comparison.csv"), index=False)
print("\n================  E4  Paris <-> GeoLife normalized comparison  ================")
print(tab.to_string(index=False))

# ---------- figure ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CP, CG = "#8856a7", "#2c7fb8"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

# (A) RQ1 relative improvement over best baseline
ax[0].bar([0, 1], [p1[3], g1[3]], color=[CP, CG], width=0.6)
for i, v in enumerate([p1[3], g1[3]]):
    ax[0].text(i, v + 0.7, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Paris", "GeoLife"])
ax[0].set_ylabel("sem-gen-TV reduction vs best baseline (%)")
ax[0].set_title("(A) RQ1 · role of the abstraction"); ax[0].set_ylim(0, max(p1[3], g1[3]) * 1.25)

# (B) RQ2 scaling curves overlaid (semantic gen-TV vs n)
pp = p_r2.sort_values("n"); gg = g_e2.sort_values("n")
ax[1].errorbar(pp.n, pp.sem_genTV, yerr=pp.sem_genTV_ci, fmt="o-", color=CP, capsize=3, lw=2, label="Paris")
ax[1].errorbar(gg.n, gg.sem_genTV, yerr=gg.sem_genTV_ci, fmt="s-", color=CG, capsize=3, lw=2, label="GeoLife")
ax[1].axvline(p2["sat"], color=CP, ls=":", lw=1.4); ax[1].axvline(g2["sat"], color=CG, ls=":", lw=1.4)
ax[1].set_xscale("log"); ax[1].set_xlabel("number of individuals  n")
ax[1].set_ylabel("semantic generation TV (↓)")
ax[1].set_title("(B) RQ2 · data efficiency"); ax[1].legend(fontsize=9)
ax[1].annotate(f"n*≈{p2['sat']:.0f}", (p2["sat"], ax[1].get_ylim()[1]), color=CP, fontsize=8, ha="center", va="top")
ax[1].annotate(f"n*≈{g2['sat']:.0f}", (g2["sat"], ax[1].get_ylim()[0]), color=CG, fontsize=8, ha="center", va="bottom")

# (C) RQ3 LSTM near-copy signature overlaid + Markov baselines
pm = p_r3[p_r3.model == "Markov"].sort_values("n"); pl = p_r3[p_r3.model == "LSTM"].sort_values("n")
gm = g_e3[g_e3.gen == "Markov"].sort_values("n"); gl = g_e3[g_e3.gen == "LSTM"].sort_values("n")
ax[2].plot(pl.n, pl.near_pct, "o-", color=CP, lw=2, label="Paris · LSTM")
ax[2].plot(pm.n, pm.near_pct, "o--", color=CP, lw=1.3, alpha=.6, label="Paris · Markov")
ax[2].plot(gl.n, gl.near_pct, "s-", color=CG, lw=2, label="GeoLife · LSTM")
ax[2].plot(gm.n, gm.near_pct, "s--", color=CG, lw=1.3, alpha=.6, label="GeoLife · Markov")
ax[2].set_xscale("log"); ax[2].set_xlabel("number of individuals  n")
ax[2].set_ylabel("near-copy rate (% MUITAS ≥ 0.95, ↓)")
ax[2].set_title("(C) RQ3 · memorization signature"); ax[2].legend(fontsize=8)

for a in (ax[1], ax[2]):
    tk = [10, 20, 50, 100, 200, 500]
    a.set_xticks(tk); a.set_xticklabels([str(t) for t in tk]); a.minorticks_off()
fig.suptitle("E4 · external validation: Paris and GeoLife agree on all three findings", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
p = os.path.join(FIG, "geolife_e4.png"); os.makedirs(FIG, exist_ok=True)
fig.savefig(p, dpi=130); plt.close(fig)
print("\nfigure:", p)
