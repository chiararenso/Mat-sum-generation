"""
RQ1 beyond bigrams: does the Grid / Cluster / MAT-Sum ranking survive when fidelity is measured on
higher-order n-grams?

An order-1 Markov chain is fitted on bigram statistics, so a low bigram TV is expected by
construction. n >= 3 is NOT estimated by the generator, so n-gram TV for n = 3, 4, 5 probes sequential
structure that neither the generator nor the representation was explicitly fitted to reproduce.

Setup is identical to val_abstraction_ci.py / val_abstraction_smoothing.py (same fixed ~328-state
abstractions, the SAME R=20 paired user samples via identical RNG seeding, alpha = 0.1), plus a
Top1-label-Markov reference (single dominant OSM label per state, ~59 states).

For every n we report, in the common OSM-label space (all methods vs the same ground truth):
  TV_n(synthetic labels, real GT labels)          [lower = better]
and a split-half real-vs-real noise floor: TV_n between two disjoint random halves of the SAME
paired real sample (each ~ the size of the synthetic release). A synthetic TV at the floor means
"as close to the real distribution as real data is to itself"; beyond the floor n-grams are
sampling-noise dominated (sparse support), so comparisons there are not interpretable.
Native-state TV (each method vs its own real distribution) is also reported for n = 2..4.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]
TARGET, N_SYNTH = 328, 200
R = 20
ALPHA = 0.1
NS_LABEL = [1, 2, 3, 4, 5]
NS_NATIVE = [2, 3, 4]
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


def ngram(seqs, n):
    c = Counter()
    for v in seqs:
        for i in range(len(v) - n + 1):
            c[tuple(v[i:i + n])] += 1
    tot = sum(c.values()) or 1
    return {k: x / tot for k, x in c.items()}


def markov_gen(real_seqs, n, grng, alpha):
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real_seqs:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = list(states)
    lengths = [len(v) for v in real_seqs if len(v) > 0] or [1]
    V = len(states)

    def nxt(state):
        c = np.array([trans[state][s] for s in states], float)
        denom = c.sum() + alpha * V
        p = (c + alpha) / denom if denom > 0 else np.ones(V) / V
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

    methods = [("Grid-Markov", "grid"), ("Cluster-Markov", "cluster"),
               ("MAT-Sum-Markov", "matsum"), ("Top1-Markov", "top1")]
    dom = {col: sem.groupby(col)["top1"].agg(lambda s: s.value_counts().index[0]).to_dict()
           for _, col in methods}
    tids = sem["tid"].unique()
    seqs_by = {col: {} for _, col in methods}; gt_by = {}
    for tid, gtr in sem.groupby("tid", sort=False):
        gt_by[tid] = rle(gtr["top1"].tolist())
        for _, col in methods:
            seqs_by[col][tid] = rle(gtr[col].tolist())

    n_users = int(len(tids) * 0.70)
    print(f"[repeat] R={R} paired samples of {n_users}/{len(tids)} users, alpha={ALPHA}, "
          f"label n={NS_LABEL}, native n={NS_NATIVE}")
    lab = {m[0]: {n: [] for n in NS_LABEL} for m in methods}
    nat = {m[0]: {n: [] for n in NS_NATIVE} for m in methods}
    floor_lab = {n: [] for n in NS_LABEL}
    floor_nat = {m[0]: {n: [] for n in NS_NATIVE} for m in methods}
    ngram_support = {n: [] for n in NS_LABEL}     # distinct real label n-grams in the sample (sparsity)

    for r in range(R):
        sample = rng.choice(tids, n_users, replace=False)
        grng = np.random.default_rng(100 + r)
        frng = np.random.default_rng(5000 + r)
        gt_seqs = [gt_by[t] for t in sample]
        gt_ng = {n: ngram(gt_seqs, n) for n in NS_LABEL}
        for n in NS_LABEL:
            ngram_support[n].append(len(gt_ng[n]))

        perm = frng.permutation(sample)
        half = len(perm) // 2
        A, B = perm[:half], perm[half:2 * half]
        for n in NS_LABEL:
            floor_lab[n].append(tv(ngram([gt_by[t] for t in A], n), ngram([gt_by[t] for t in B], n)))

        for name, col in methods:
            real = [seqs_by[col][t] for t in sample]
            synth = markov_gen(real, N_SYNTH, grng, ALPHA)
            synth_lab = [rle([dom[col][s] for s in seq]) for seq in synth]
            for n in NS_LABEL:
                lab[name][n].append(tv(ngram(synth_lab, n), gt_ng[n]))
            for n in NS_NATIVE:
                nat[name][n].append(tv(ngram(synth, n), ngram(real, n)))
                floor_nat[name][n].append(
                    tv(ngram([seqs_by[col][t] for t in A], n), ngram([seqs_by[col][t] for t in B], n)))
        print(f"  rep {r + 1}/{R} done", flush=True)

    def mci(a):
        a = np.asarray(a, float)
        return float(a.mean()), float(TCRIT * a.std(ddof=1) / np.sqrt(len(a)))

    rows = []
    for n in NS_LABEL:
        m, c = mci(floor_lab[n])
        rows.append({"space": "label", "n": n, "method": "real-vs-real floor", "mean": round(m, 4), "ci95": round(c, 4)})
        for name, _ in methods:
            m, c = mci(lab[name][n])
            rows.append({"space": "label", "n": n, "method": name, "mean": round(m, 4), "ci95": round(c, 4)})
    for n in NS_NATIVE:
        for name, _ in methods:
            m, c = mci(nat[name][n])
            fm, _fc = mci(floor_nat[name][n])
            rows.append({"space": "native", "n": n, "method": name, "mean": round(m, 4), "ci95": round(c, 4),
                         "floor_mean": round(fm, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "val_abstraction_ngrams.csv"), index=False)
    pd.set_option("display.width", 220)

    print("\n=== common OSM-label space: TV_n (mean +/- 95% CI, R=20); lower = better ===")
    piv = df[df.space == "label"].assign(v=lambda d: d["mean"].map("{:.3f}".format) + "±" + d["ci95"].map("{:.3f}".format))
    print(piv.pivot(index="method", columns="n", values="v").to_string())
    print("\n=== native state space: TV_n (+ real-vs-real split-half floor in own states) ===")
    nd = df[df.space == "native"].assign(v=lambda d: d["mean"].map("{:.3f}".format) + "±" + d["ci95"].map("{:.3f}".format)
                                         + " [floor " + d["floor_mean"].map("{:.3f}".format) + "]")
    print(nd.pivot(index="method", columns="n", values="v").to_string())
    print("\n[distinct real label n-grams in the paired sample, mean]:",
          {n: round(float(np.mean(ngram_support[n])), 1) for n in NS_LABEL})

    print("\n=== paired significance: MAT-Sum vs baseline, label-space TV_n (d = TV_baseline - TV_MAT-Sum) ===")
    prows = []
    for n in NS_LABEL:
        for base in ("Grid-Markov", "Cluster-Markov", "Top1-Markov"):
            d = np.array(lab[base][n]) - np.array(lab["MAT-Sum-Markov"][n])
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
            wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            wins = int((d > 0).sum())
            prows.append({"n": n, "baseline": base, "d_mean": round(float(d.mean()), 4), "d_z": round(float(dz), 2),
                          "wilcoxon_p": wp, "matsum_wins_of_20": wins})
            print(f"  n={n}  {base:15s} d_mean={d.mean():+.4f}  d_z={dz:6.1f}  wilcoxon_p={wp:.2e}  wins={wins}/{R}")
    pd.DataFrame(prows).to_csv(os.path.join(OUT, "val_abstraction_ngrams_paired.csv"), index=False)
    print("\nsaved:", os.path.join(OUT, "val_abstraction_ngrams.csv"))
    print("saved:", os.path.join(OUT, "val_abstraction_ngrams_paired.csv"))


if __name__ == "__main__":
    main()
