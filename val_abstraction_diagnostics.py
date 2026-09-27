"""
Representation diagnostics for RQ1 (Paris-581), addressing the reviewer concern that a lower
native-state bigram-TV (Table "val_abstraction_ci") could reflect a more concentrated/regular
empirical distribution rather than "semantics" per se. We therefore report, per abstraction and
per paired user sample (SAME R=20 samples as val_abstraction_ci.py, same RNG seeding so the two
tables are directly comparable):

  - effective vocabulary size    : number of distinct states actually visited in the sample
  - transition-graph density     : distinct observed (i -> j) transitions / (#states)^2
  - mean/median support          : average / median real-transition count backing each observed edge
  - conditional entropy H(j|i)   : bits, Laplace-smoothed exactly like the paper's order-1 Markov
                                    generator (Sec. 3.4: (C_ij+1)/(sum_j' C_ij' + |V|)), weighted
                                    by each state's visitation frequency
  - perplexity                   : 2^H

This script does NOT touch the synthetic-generation or fidelity metrics already reported in
val_abstraction_ci.py; it only characterizes the REAL empirical state space each abstraction
induces, at the same matched ~328-state budget.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]
TARGET = 328
R, SAMPLE_FRAC = 20, 0.70
rng = np.random.default_rng(0)          # identical seed/draw order to val_abstraction_ci.py
try:
    from scipy.stats import t as tdist
    TCRIT = float(tdist.ppf(0.975, df=R - 1))
except Exception:
    TCRIT = 2.262


def rle(vals):
    out = []
    for v in vals:
        if not out or out[-1] != v:
            out.append(v)
    return out


def build_trans(real_seqs):
    """Same transition-counting convention as markov_gen() in val_abstraction_ci.py."""
    START = "<S>"
    trans = defaultdict(Counter)
    for v in real_seqs:
        prev = START
        for s in v:
            trans[prev][s] += 1
            prev = s
    return trans, START


def diagnostics(real_seqs):
    trans, START = build_trans(real_seqs)
    states = sorted(s for s in trans if s != START)
    V = len(states)
    if V == 0:
        return dict(n_states=0, n_transitions=0, density=np.nan, mean_support=np.nan,
                    median_support=np.nan, entropy_bits=np.nan, perplexity=np.nan)

    # visitation frequency of each state (as an outgoing-transition source), for weighting H
    visit_count = {s: sum(trans[s].values()) for s in states}
    total_visits = sum(visit_count.values()) or 1

    supports = []
    n_transitions = 0
    H = 0.0
    for s in states:
        c = trans[s]
        if not c:
            continue
        counts = np.array(list(c.values()), float)
        supports.extend(counts.tolist())
        n_transitions += len(counts)
        # Laplace-smoothed conditional distribution, exactly as Sec. 3.4
        p = (counts + 1) / (counts.sum() + V)
        # unobserved successors also carry probability mass 1/(sum+V); fold their entropy contribution in closed form
        p_unseen = 1.0 / (counts.sum() + V)
        n_unseen = V - len(counts)
        h_s = -(p * np.log2(p)).sum() - n_unseen * (p_unseen * np.log2(p_unseen)) if n_unseen > 0 else -(p * np.log2(p)).sum()
        w = visit_count[s] / total_visits
        H += w * h_s

    supports = np.array(supports, float)
    density = n_transitions / (V ** 2)
    return dict(
        n_states=V,
        n_transitions=n_transitions,
        density=density,
        mean_support=float(supports.mean()) if len(supports) else np.nan,
        median_support=float(np.median(supports)) if len(supports) else np.nan,
        entropy_bits=H,
        perplexity=2 ** H,
    )


def main():
    print("[load] Paris-581 + build fixed abstractions (~%d states)" % TARGET)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    xy = gpd.GeoSeries(sem.geometry, crs="EPSG:4326").to_crs(2154)
    sem["x"] = xy.x.values; sem["y"] = xy.y.values

    # MAT-Sum states
    raw = SY.build_visits(sem, 2); SY.anonymize_vocabulary(raw, 5)
    fs = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, c in fs.items() if c >= 5 and len(s) > 0]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    sem["matsum"] = sem["sig2"].map(lambda s: remap.get(s, s))
    # Grid states (tune to ~TARGET visited cells)
    lo, hi = 0.0015, 0.06
    for _ in range(30):
        g = (lo + hi) / 2
        nv = len(set(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int))))
        lo, hi = (g, hi) if nv > TARGET else (lo, g)
    g = (lo + hi) / 2
    sem["grid"] = list(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int)))
    # Cluster states
    XY = sem[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=TARGET, n_init=3, random_state=0).fit(XY[rng.choice(len(XY), 40000, replace=False)])
    sem["cluster"] = km.predict(XY)

    methods = [("Grid-Markov", "grid"), ("Cluster-Markov", "cluster"), ("MAT-Sum-Markov", "matsum")]
    tids = sem["tid"].unique()
    seqs_by = {col: {} for _, col in methods}
    for tid, gtr in sem.groupby("tid", sort=False):
        for _, col in methods:
            seqs_by[col][tid] = rle(gtr[col].tolist())

    metrics = ["n_states", "n_transitions", "density", "mean_support", "median_support",
               "entropy_bits", "perplexity"]
    acc = {m[0]: {k: [] for k in metrics} for m in methods}
    n_users = int(len(tids) * SAMPLE_FRAC)
    print(f"[repeat] R={R} paired samples of {n_users}/{len(tids)} users (identical draws to val_abstraction_ci.py)")
    for r in range(R):
        sample = rng.choice(tids, n_users, replace=False)
        for name, col in methods:
            real = [seqs_by[col][t] for t in sample]
            d = diagnostics(real)
            for k in metrics:
                acc[name][k].append(d[k])
        print(f"  rep {r + 1}/{R} done", flush=True)

    rows = []
    for name, _ in methods:
        row = {"method": name}
        for k in metrics:
            a = np.array(acc[name][k], float)
            ci = TCRIT * a.std(ddof=1) / np.sqrt(len(a))
            row[f"{k}_mean"] = round(float(a.mean()), 4)
            row[f"{k}_ci95"] = round(float(ci), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    res_dir = os.path.join(OUT.replace("/data/paris", "/results/paris") if "/data/paris" in OUT
                            else (OUT.replace("/output", "/results/paris") if "/output" in OUT else OUT))
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, "val_abstraction_diagnostics.csv")
    df.to_csv(out_path, index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))
    print("\nsaved:", out_path)

    # ---- raw per-replicate values (for paired testing / reuse) ----
    raw_rows = []
    for r in range(R):
        for name, _ in methods:
            raw_rows.append({"rep": r, "method": name, **{k: acc[name][k][r] for k in metrics}})
    raw_path = os.path.join(res_dir, "val_abstraction_diagnostics_raw.csv")
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    print("saved:", raw_path)

    # ---- paired significance: d_i = baseline - MAT-Sum, same protocol as val_abstraction_ci.py ----
    from scipy.stats import wilcoxon, shapiro, ttest_1samp
    brng = np.random.default_rng(7)

    def analyse(d):
        d = np.asarray(d, float); n = len(d)
        boot = np.array([brng.choice(d, n, replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
        B = 20000; signs = brng.choice([-1, 1], size=(B, n))
        pm = (np.abs((signs * np.abs(d)).mean(1)) >= abs(d.mean())).mean()
        pm = (pm * B + 1) / (B + 1)
        wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        sp = float(shapiro(d).pvalue) if n >= 3 else float("nan")
        tp = float(ttest_1samp(d, 0).pvalue)
        return dict(mean=d.mean(), median=np.median(d), ci_lo=lo, ci_hi=hi, dz=dz,
                    p_perm=pm, p_wilcoxon=wp, p_shapiro=sp, p_ttest=tp)

    ms = "MAT-Sum-Markov"; prows = []
    # positive d = MAT-Sum has MORE states/transitions/density/support than baseline;
    # for entropy_bits / perplexity, positive d = MAT-Sum has LOWER (baseline - MAT-Sum > 0 = more regular)
    for base in ("Grid-Markov", "Cluster-Markov"):
        for k in metrics:
            if k in ("entropy_bits", "perplexity"):
                d = np.array(acc[base][k]) - np.array(acc[ms][k])   # >0 = MAT-Sum more regular
            else:
                d = np.array(acc[ms][k]) - np.array(acc[base][k])  # >0 = MAT-Sum higher
            st = analyse(d)
            prows.append({"baseline": base, "metric": k, **{kk: round(vv, 4) for kk, vv in st.items()}})
    pdf = pd.DataFrame(prows)
    paired_path = os.path.join(res_dir, "val_abstraction_diagnostics_paired.csv")
    pdf.to_csv(paired_path, index=False)
    print(f"\n[paired] MAT-Sum vs baseline, R={R} (see column convention in code comments)")
    for _, r in pdf.iterrows():
        print(f"  {r.baseline:15s} {r.metric:16s}  d mean={r['mean']:+.4f} median={r['median']:+.4f}  "
              f"CI95=[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  d_z={r.dz:.1f}  "
              f"p_perm={r.p_perm:.1e} p_wilcox={r.p_wilcoxon:.1e} (Shapiro={r.p_shapiro:.2f})")
    print("saved:", paired_path)


if __name__ == "__main__":
    main()
