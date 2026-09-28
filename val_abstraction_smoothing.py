"""
Sensitivity of the RQ1 abstraction ranking (Grid vs Cluster vs MAT-Sum) to the Laplace
smoothing strength alpha, addressing the reviewer concern that add-one-style smoothing over a
large alphabet (|V| ~ 328) could disproportionately damage sparser representations (Grid/Cluster)
relative to MAT-Sum's denser transition graph (Table 7), rather than the ranking reflecting an
intrinsically easier transition process.

P(j|i) = (C_ij + alpha) / (sum_j' C_ij' + alpha*|V|)

Same setup, same fixed ~328-state abstractions, and the SAME R=20 paired user samples (identical
RNG seeding) as val_abstraction_ci.py (alpha=0.1, the paper's actual default -- NOT alpha=1 as an
earlier version of the paper's formula stated; this script also serves to confirm that at
alpha=0.1 it reproduces Table 4/5's published numbers). We additionally sweep
alpha in {0, 0.01, 0.1, 1}.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]
TARGET, N_SYNTH = 328, 200
R = 20
ALPHAS = [0.0, 0.01, 0.1, 1.0]
rng = np.random.default_rng(0)          # identical seed/draw order to val_abstraction_ci.py
try:
    from scipy.stats import t as tdist, wilcoxon
    TCRIT = float(tdist.ppf(0.975, R - 1))
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


def bigram(seqs):
    c = Counter()
    for v in seqs:
        for a, b in zip(v, v[1:]):
            c[(a, b)] += 1
    n = sum(c.values()) or 1
    return {k: x / n for k, x in c.items()}


def markov_gen(real_seqs, n, grng, alpha):
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real_seqs:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = list(states); idx = {s: i for i, s in enumerate(states)}
    lengths = [len(v) for v in real_seqs if len(v) > 0] or [1]
    V = len(states)
    def nxt(state):
        c = np.array([trans[state][s] for s in states], float)
        denom = c.sum() + alpha * V
        if denom <= 0:
            p = np.ones(V) / V
        else:
            p = (c + alpha) / denom
        return states[grng.choice(V, p=p)]
    out = []
    for _ in range(n):
        L = int(grng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            st = nxt(st); seq.append(st)
        out.append(seq)
    return out


def main():
    print("[load] Paris-581 + build fixed abstractions (~%d states)" % TARGET)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    xy = gpd.GeoSeries(sem.geometry, crs="EPSG:4326").to_crs(2154)
    sem["x"] = xy.x.values; sem["y"] = xy.y.values

    raw = SY.build_visits(sem, 2); SY.anonymize_vocabulary(raw, 5)
    fs = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, c in fs.items() if c >= 5 and len(s) > 0]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    sem["matsum"] = sem["sig2"].map(lambda s: remap.get(s, s))
    lo, hi = 0.0015, 0.06
    for _ in range(30):
        g = (lo + hi) / 2
        nv = len(set(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int))))
        lo, hi = (g, hi) if nv > TARGET else (lo, g)
    g = (lo + hi) / 2
    sem["grid"] = list(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int)))
    XY = sem[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=TARGET, n_init=3, random_state=0).fit(XY[rng.choice(len(XY), 40000, replace=False)])
    sem["cluster"] = km.predict(XY)

    methods = [("Grid-Markov", "grid"), ("Cluster-Markov", "cluster"), ("MAT-Sum-Markov", "matsum")]
    dom = {col: sem.groupby(col)["top1"].agg(lambda s: s.value_counts().index[0]).to_dict()
           for _, col in methods}
    tids = sem["tid"].unique()
    seqs_by = {col: {} for _, col in methods}; gt_by = {}
    for tid, gtr in sem.groupby("tid", sort=False):
        gt_by[tid] = rle(gtr["top1"].tolist())
        for _, col in methods:
            seqs_by[col][tid] = rle(gtr[col].tolist())

    n_users = int(len(tids) * 0.70)
    print(f"[repeat] R={R} paired samples of {n_users}/{len(tids)} users, alphas={ALPHAS}")
    metrics = ["generic_bigramTV", "semantic_genTV", "exact_copies"]
    acc = {a: {m[0]: {k: [] for k in metrics} for m in methods} for a in ALPHAS}
    for r in range(R):
        sample = rng.choice(tids, n_users, replace=False)
        gt_bi = bigram([gt_by[t] for t in sample])
        for alpha in ALPHAS:
            grng = np.random.default_rng(100 + r)  # same per-(r) seed across alphas -> paired across alpha too
            for name, col in methods:
                real = [seqs_by[col][t] for t in sample]
                synth = markov_gen(real, N_SYNTH, grng, alpha)
                real_lab = [rle([dom[col][s] for s in seq]) for seq in real]
                synth_lab = [rle([dom[col][s] for s in seq]) for seq in synth]
                real_tuples = set(tuple(s) for s in real)
                acc[alpha][name]["generic_bigramTV"].append(tv(bigram(synth), bigram(real)))
                acc[alpha][name]["semantic_genTV"].append(tv(bigram(synth_lab), gt_bi))
                acc[alpha][name]["exact_copies"].append(sum(tuple(s) in real_tuples for s in synth))
        print(f"  rep {r+1}/{R} done", flush=True)

    rows = []
    for alpha in ALPHAS:
        for name, _ in methods:
            row = {"alpha": alpha, "method": name}
            for k in metrics:
                a = np.array(acc[alpha][name][k], float)
                ci = TCRIT * a.std(ddof=1) / np.sqrt(len(a))
                row[f"{k}_mean"] = round(float(a.mean()), 4)
                row[f"{k}_ci95"] = round(float(ci), 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "val_abstraction_smoothing.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))

    # ranking check + paired significance (MAT-Sum vs each baseline) per alpha
    print("\n[ranking + paired significance per alpha] (MAT-Sum vs baseline, semantic_genTV, R=20)")
    prows = []
    for alpha in ALPHAS:
        order = sorted(methods, key=lambda m: np.mean(acc[alpha][m[0]]["semantic_genTV"]))
        ranking = " < ".join(m[0] for m, _ in [(o, None) for o in order])
        print(f"  alpha={alpha}: ranking by semantic_genTV (best first): {ranking}")
        ms = "MAT-Sum-Markov"
        for base in ("Grid-Markov", "Cluster-Markov"):
            d = np.array(acc[alpha][base]["semantic_genTV"]) - np.array(acc[alpha][ms]["semantic_genTV"])
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
            wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            prows.append({"alpha": alpha, "baseline": base, "d_mean": round(float(d.mean()), 4),
                          "d_z": round(float(dz), 2), "wilcoxon_p": wp})
            print(f"    {base:15s}: d_mean={d.mean():+.4f}  d_z={dz:.1f}  wilcoxon_p={wp:.2e}")
    pd.DataFrame(prows).to_csv(os.path.join(OUT, "val_abstraction_smoothing_paired.csv"), index=False)
    print("\nsaved:", os.path.join(OUT, "val_abstraction_smoothing.csv"))
    print("saved:", os.path.join(OUT, "val_abstraction_smoothing_paired.csv"))


if __name__ == "__main__":
    main()
