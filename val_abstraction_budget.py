"""
Advantage-at-matched-budget (Paris-581)  --  kills the "semantics wins only because it is
a smaller/simpler vocabulary" objection.

For a RANGE of state budgets B, we build THREE abstractions each with (approximately) B states:
  * Grid-Markov     : geometric grid tuned to ~B visited cells
  * Cluster-Markov  : KMeans on projected coordinates with B clusters
  * MAT-Sum-Markov  : top-2 semantic signatures, codebook = the B most frequent, the rest
                      remapped to their nearest kept signature by Jaccard (same merge mechanism
                      as the (m,k) vocabulary, just with the support threshold set to hit B).
All three are driven by the SAME order-1 Markov and evaluated in the SAME common OSM-label space
(semantic generation TV, lower = better) plus exact state-sequence copies.  R paired user samples
per budget -> mean +/- 95% CI.

If semantics still wins at EVERY matched B, the win cannot come from having fewer/simpler states.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
BUDGETS = [40, 80, 160, 320, 480]
N_SYNTH = 200
R, SAMPLE_FRAC = 20, 0.70
rng = np.random.default_rng(0)
try:
    from scipy.stats import t as tdist
    TCRIT = float(tdist.ppf(0.975, df=R - 1))
except Exception:
    TCRIT = 2.093


def rle(vals):
    out = []
    for v in vals:
        if not out or out[-1] != v:
            out.append(v)
    return out


def tv(pa, pb):
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))


def ngram(seqs, k):
    c = Counter()
    for v in seqs:
        for i in range(len(v) - k + 1):
            c[tuple(v[i:i + k])] += 1
    n = sum(c.values()) or 1
    return {t: x / n for t, x in c.items()}


def bigram(seqs):
    return ngram(seqs, 2)


def markov_gen(real_seqs, n, grng):
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real_seqs:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = list(states)
    lengths = [len(v) for v in real_seqs if len(v) > 0] or [1]
    def nxt(state):
        c = np.array([trans[state][s] for s in states], float)
        return states[grng.choice(len(states), p=(c + 0.1) / (c.sum() + 0.1 * len(states)))]
    out = []
    for _ in range(n):
        L = int(grng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            st = nxt(st); seq.append(st)
        out.append(seq)
    return out


def build_matsum_col(sem, fs, B):
    """codebook = B most frequent top-2 signatures; remap the rest to nearest by Jaccard."""
    keep = [s for s, _ in fs.most_common(B) if len(s) > 0]
    kset = set(keep)
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in kset else (max(keep, key=lambda f: jac(s, f)) if keep else s)) for s in fs}
    return sem["sig2"].map(lambda s: remap.get(s, s))


def build_grid_col(sem, B):
    lo, hi = 0.0008, 0.08
    for _ in range(34):
        g = (lo + hi) / 2
        nv = len(set(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int))))
        lo, hi = (g, hi) if nv > B else (lo, g)
    g = (lo + hi) / 2
    col = list(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int)))
    nv = len(set(col))
    return col, nv


def main():
    print("[load] Paris-581", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    xy = gpd.GeoSeries(sem.geometry, crs="EPSG:4326").to_crs(2154)
    sem["x"] = xy.x.values; sem["y"] = xy.y.values
    raw = SY.build_visits(sem, 2)
    fs = Counter(s for v in raw.values() for s, _ in v)
    tids = sem["tid"].unique()
    n_users = int(len(tids) * SAMPLE_FRAC)

    # GT label sequences per tid (common OSM-label space) -- fixed across everything
    gt_by = {tid: rle(g["top1"].tolist()) for tid, g in sem.groupby("tid", sort=False)}

    XY = sem[["x", "y"]].to_numpy()
    sub = XY[rng.choice(len(XY), min(30000, len(XY)), replace=False)]

    methods = ["Grid-Markov", "Cluster-Markov", "MAT-Sum-Markov"]
    OP = 320                                   # operating point at which the Top1 ablation is paired
    n_top1 = int(sem["top1"].nunique())        # trivial single-label alphabet size
    records = []
    top1_acc = {"genTV2": [], "genTV3": [], "copies": []}
    for B in BUDGETS:
        print(f"\n[budget] target B={B}", flush=True)
        cols = {}
        sem["matsum"] = build_matsum_col(sem, fs, B); cols["MAT-Sum-Markov"] = "matsum"
        gcol, nv = build_grid_col(sem, B); sem["grid"] = gcol; cols["Grid-Markov"] = "grid"
        km = KMeans(n_clusters=B, n_init=2, random_state=0).fit(sub)
        sem["cluster"] = km.predict(XY); cols["Cluster-Markov"] = "cluster"
        realized = {"Grid-Markov": nv,
                    "Cluster-Markov": B,
                    "MAT-Sum-Markov": int(sem["matsum"].nunique())}
        print("   realized #states:", realized, flush=True)

        dom = {c: sem.groupby(c)["top1"].agg(lambda s: s.value_counts().index[0]).to_dict()
               for c in cols.values()}
        seqs_by = {c: {} for c in cols.values()}
        for tid, g in sem.groupby("tid", sort=False):
            for c in cols.values():
                seqs_by[c][tid] = rle(g[c].tolist())

        acc = {m: {"genTV2": [], "genTV3": [], "copies": []} for m in methods}
        for r in range(R):
            sample = rng.choice(tids, n_users, replace=False)
            grng = np.random.default_rng(1000 * B + r)
            gt_seqs = [gt_by[t] for t in sample]
            gt_bi2, gt_bi3 = ngram(gt_seqs, 2), ngram(gt_seqs, 3)
            for m in methods:
                c = cols[m]
                real = [seqs_by[c][t] for t in sample]
                synth = markov_gen(real, N_SYNTH, grng)
                synth_lab = [rle([dom[c][s] for s in seq]) for seq in synth]
                real_tuples = set(tuple(s) for s in real)
                acc[m]["genTV2"].append(tv(ngram(synth_lab, 2), gt_bi2))
                acc[m]["genTV3"].append(tv(ngram(synth_lab, 3), gt_bi3))
                acc[m]["copies"].append(sum(tuple(s) in real_tuples for s in synth))
            # ---- Top1 ablation: trivial single-label summarization, SAME sample, at operating point
            if B == OP:
                g2 = np.random.default_rng(777000 + r)
                real1 = gt_seqs                                   # sequence of dominant OSM labels
                synth1 = markov_gen(real1, N_SYNTH, g2)           # Markov directly in label space
                rt1 = set(tuple(s) for s in real1)
                top1_acc["genTV2"].append(tv(ngram(synth1, 2), gt_bi2))
                top1_acc["genTV3"].append(tv(ngram(synth1, 3), gt_bi3))
                top1_acc["copies"].append(sum(tuple(s) in rt1 for s in synth1))
            if (r + 1) % 5 == 0:
                print(f"   rep {r+1}/{R}", flush=True)

        for m in methods:
            for k in ("genTV2", "genTV3", "copies"):
                a = np.array(acc[m][k], float)
                ci = TCRIT * a.std(ddof=1) / np.sqrt(len(a))
                records.append({"budget": B, "states": realized[m], "method": m, "metric": k,
                                "mean": float(a.mean()), "ci95": float(ci)})
    # Top1 aggregate (single point at its native alphabet size)
    top1 = {}
    for k in ("genTV2", "genTV3", "copies"):
        a = np.array(top1_acc[k], float)
        top1[k] = (float(a.mean()), float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))))
        records.append({"budget": OP, "states": n_top1, "method": "Top1-label-Markov",
                        "metric": k, "mean": top1[k][0], "ci95": top1[k][1]})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT, "val_abstraction_budget.csv"), index=False)
    pd.set_option("display.width", 200)
    for metric, lab in [("genTV2", "semantic gen-TV bigram"), ("genTV3", "semantic gen-TV trigram"),
                        ("copies", "exact copies (of 200)")]:
        print(f"\n=== {lab} (lower = better) ===")
        print(df[df.metric == metric].pivot_table(index="budget", columns="method",
              values="mean").round(3).to_string())
    print(f"\n=== Top1-label-Markov ablation (trivial single-label, {n_top1} states, paired @ B={OP}) ===")
    print(f"   bigram gen-TV  {top1['genTV2'][0]:.3f} ± {top1['genTV2'][1]:.3f}")
    print(f"   trigram gen-TV {top1['genTV3'][0]:.3f} ± {top1['genTV3'][1]:.3f}")
    print(f"   exact copies   {top1['copies'][0]:.2f} ± {top1['copies'][1]:.2f}")

    # ---- figure (3 panels) ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    style = {"Grid-Markov": ("#8aa0b3", "o"),
             "Cluster-Markov": ("#2c7fb8", "s"),
             "MAT-Sum-Markov": ("#d95f0e", "D")}
    T1C = "#6a3d9a"
    panels = [("genTV2", "(A) Fidelity — bigram (label space)\nTop1 is native to this metric (see text)",
               "semantic gen-TV, bigram  (↓ better)"),
              ("genTV3", "(B) Fidelity — trigram (label space)\nsame caveat for Top1",
               "semantic gen-TV, trigram  (↓ better)"),
              ("copies", "(C) Memorization — MAT-Sum's real edge\nfine resolution at ≈0 copies; Top1 is capped coarse",
               "exact copies of real records  (of 200)")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (metric, title, ylab) in zip(axes, panels):
        for m in methods:
            d = df[(df.method == m) & (df.metric == metric)].sort_values("states")
            col, mk = style[m]
            ax.errorbar(d["states"], d["mean"], yerr=d["ci95"], marker=mk, ls="-", color=col,
                        capsize=3, lw=2, ms=6, label=m)
        t = df[(df.method == "Top1-label-Markov") & (df.metric == metric)]
        ax.errorbar(t["states"], t["mean"], yerr=t["ci95"], marker="*", ls="none", color=T1C,
                    capsize=3, ms=16, label="Top1-label-Markov (trivial)")
        ax.set_xscale("log")
        ax.set_xlabel("number of states  (matched budget, log)")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Advantage at matched budget · Paris-581 · same order-1 Markov · %d paired samples (95%% CI)" % R,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "val_abstraction_budget.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
