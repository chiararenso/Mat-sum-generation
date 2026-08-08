"""
Role of the abstraction with error bars (Paris-581).
The three ~328-state abstractions (Grid / Cluster / MAT-Sum) are built ONCE as a fixed codebook
and driven by the same order-1 Markov. We then repeat the evaluation over R paired user samples:
each repetition draws one subset of users used IDENTICALLY by the three methods, and we report
mean and 95% confidence interval (Student's t) for each metric.

Metrics (TV, lower is better): generic state-bigram fidelity; semantic distortion (real states ->
dominant OSM label bigram vs ground truth); semantic generation (synth -> label bigram vs GT);
plus exact state-sequence copies of the sampled real set.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
TARGET, N_SYNTH = 328, 200
R, SAMPLE_FRAC = 20, 0.70          # repetitions, 70% user subsample each
rng = np.random.default_rng(0)
try:
    from scipy.stats import t as tdist
    TCRIT = float(tdist.ppf(0.975, df=R - 1))
except Exception:
    TCRIT = 2.262                   # t_{0.975, 9}


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
    # fixed codebook: per-state dominant label; per-tid state/label sequences
    dom = {col: sem.groupby(col)["top1"].agg(lambda s: s.value_counts().index[0]).to_dict()
           for _, col in methods}
    tids = sem["tid"].unique()
    seqs_by = {col: {} for _, col in methods}; gt_by = {}
    for tid, gtr in sem.groupby("tid", sort=False):
        gt_by[tid] = rle(gtr["top1"].tolist())
        for _, col in methods:
            seqs_by[col][tid] = rle(gtr[col].tolist())

    # ---- repetitions over paired user samples ----
    metrics = ["generic_bigramTV", "semantic_distortion", "semantic_genTV", "exact_copies"]
    acc = {m[0]: {k: [] for k in metrics} for m in methods}
    n_users = int(len(tids) * SAMPLE_FRAC)
    print(f"[repeat] R={R} paired samples of {n_users}/{len(tids)} users")
    for r in range(R):
        sample = rng.choice(tids, n_users, replace=False)            # same sample for all 3 methods
        grng = np.random.default_rng(100 + r)
        gt_bi = bigram([gt_by[t] for t in sample])
        for name, col in methods:
            real = [seqs_by[col][t] for t in sample]
            synth = markov_gen(real, N_SYNTH, grng)
            real_lab = [rle([dom[col][s] for s in seq]) for seq in real]
            synth_lab = [rle([dom[col][s] for s in seq]) for seq in synth]
            real_tuples = set(tuple(s) for s in real)
            acc[name]["generic_bigramTV"].append(tv(bigram(synth), bigram(real)))
            acc[name]["semantic_distortion"].append(tv(bigram(real_lab), gt_bi))
            acc[name]["semantic_genTV"].append(tv(bigram(synth_lab), gt_bi))
            acc[name]["exact_copies"].append(sum(tuple(s) in real_tuples for s in synth))
        print(f"  rep {r+1}/{R} done", flush=True)

    # ---- aggregate: mean +/- 95% CI ----
    rows = []
    for name, _ in methods:
        row = {"method": name}
        for k in metrics:
            a = np.array(acc[name][k], float)
            ci = TCRIT * a.std(ddof=1) / np.sqrt(len(a))
            row[f"{k}_mean"] = round(float(a.mean()), 3)
            row[f"{k}_ci95"] = round(float(ci), 3)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "val_abstraction_ci.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    for name, _ in methods:
        print(f"\n{name}:")
        for k in metrics:
            print(f"   {k:22s} {df[df.method==name][k+'_mean'].iloc[0]:.3f}  ± {df[df.method==name][k+'_ci95'].iloc[0]:.3f} (95% CI)")

    # ---- paired per-sample improvement:  d_i = TV_{i,baseline} - TV_{i,MAT-Sum} ----
    from scipy.stats import wilcoxon, shapiro, ttest_1samp
    brng = np.random.default_rng(7)
    def analyse(d):
        d = np.asarray(d, float); n = len(d)
        # bootstrap 95% CI of the mean difference
        boot = np.array([brng.choice(d, n, replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        dz = d.mean() / d.std(ddof=1)                              # Cohen's d_z (paired)
        # exact-ish paired sign-flip permutation test (two-sided)
        B = 20000; signs = brng.choice([-1, 1], size=(B, n))
        pm = (np.abs((signs * np.abs(d)).mean(1)) >= abs(d.mean())).mean()
        pm = (pm * B + 1) / (B + 1)
        wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        sp = float(shapiro(d).pvalue) if n >= 3 else float("nan")
        tp = float(ttest_1samp(d, 0).pvalue)
        return dict(mean=d.mean(), median=np.median(d), ci_lo=lo, ci_hi=hi, dz=dz,
                    p_perm=pm, p_wilcoxon=wp, p_shapiro=sp, p_ttest=tp)

    ms = "MAT-Sum-Markov"; prows = []
    for base in ("Grid-Markov", "Cluster-Markov"):
        for k in ("generic_bigramTV", "semantic_distortion", "semantic_genTV"):
            st = analyse(np.array(acc[base][k]) - np.array(acc[ms][k]))
            prows.append({"baseline": base, "metric": k, **{kk: round(vv, 4) for kk, vv in st.items()}})
    pdf = pd.DataFrame(prows); pdf.to_csv(os.path.join(OUT, "val_abstraction_paired.csv"), index=False)
    print(f"\n[paired improvement]  d_i = TV_baseline - TV_MAT-Sum  (>0 = MAT-Sum better), R={R}")
    for _, r in pdf.iterrows():
        print(f"  {r.baseline:15s} {r.metric:20s}  d mean={r['mean']:+.3f} median={r['median']:+.3f}  "
              f"CI95=[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  d_z={r.dz:.1f}  "
              f"p_perm={r.p_perm:.1e} p_wilcox={r.p_wilcoxon:.1e} (Shapiro={r.p_shapiro:.2f})")

    # ---- figure with error bars ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(len(df))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.7))
    # Panel A: NATIVE-state bigram TV -- each method vs its OWN ~328-state real distribution
    axA.bar(x, df["generic_bigramTV_mean"], 0.55, yerr=df["generic_bigramTV_ci95"],
            capsize=4, color="#2c7fb8")
    axA.set_xticks(x); axA.set_xticklabels(df.method, rotation=12, fontsize=9)
    axA.set_ylabel("TV distance  (↓ better)")
    axA.set_title("(A) native-state bigram TV\n(each method in its own state space)", fontsize=11)
    # Panel B: COMMON-label-space TV -- all methods vs the SAME OSM-label ground truth
    w = 0.38
    axB.bar(x - w/2, df["semantic_distortion_mean"], w, yerr=df["semantic_distortion_ci95"],
            capsize=4, color="#8aa0b3", label="abstraction distortion (real→labels)")
    axB.bar(x + w/2, df["semantic_genTV_mean"], w, yerr=df["semantic_genTV_ci95"],
            capsize=4, color="#d95f0e", label="generation (synth→labels)")
    axB.set_xticks(x); axB.set_xticklabels(df.method, rotation=12, fontsize=9)
    axB.set_title("(B) common-label-space TV\n(all vs the same OSM-label ground truth)", fontsize=11)
    axB.legend(fontsize=8.5)
    for a in (axA, axB):
        a.set_ylim(0, 0.72)
    fig.suptitle(f"Role of the abstraction · Paris-581 · ~{TARGET} states · {R} paired samples (95% CI)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "val_abstraction_ci.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
