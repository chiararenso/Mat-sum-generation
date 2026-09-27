"""
Downstream demonstration of the added value of the multi-aspect (m=2) representation over a
trivial single-label (Top1) release, addressing the reviewer objection: "if Top1-label-Markov
wins on fidelity, why not just release single labels?"

Query: mixed-use structure. Many MAT-Sum semantic locations carry TWO aspects at m=2
(e.g. {accommodation, food_and_beverages}), not just their dominant one. For each dominant
OSM label c (e.g. "accommodation"), we ask: what is the distribution of the SECONDARY aspect
among visits to c (including "NONE" for genuinely single-use locations)? This is a standard
urban-planning-relevant mixed-use statistic that a Top1-only release cannot express at all,
because it never retains a second aspect.

We report, over R=20 paired user samples (identical draws to val_abstraction_ci.py):
  (1) baseline_JSD  = JSD( P(secondary | dominant=c) , P(secondary) )  on the REAL data,
      weighted by each category's real visit frequency. This quantifies how much
      category-specific mixed-use information EXISTS to be lost if only Top1 labels are kept
      (a property of the representation/data, not of any generator).
  (2) synth_JSD     = JSD( P(secondary | dominant=c) on REAL , same on MAT-SUM-MARKOV SYNTHETIC ),
      same weighting. This quantifies how well the multi-aspect synthetic release preserves that
      structure.
A large gap (baseline_JSD >> synth_JSD) demonstrates concretely that (i) the multi-aspect context
carries real, non-trivial information beyond the dominant label, and (ii) the MAT-Sum-based
release recovers most of it, whereas a Top1-only consumer is stuck with the context-free
marginal regardless of generator quality.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
import matsum_summarize as MS

OUT = os.environ["MATSUM_OUT"]
TARGET_K = 5          # MAT-Sum operating point k (m=2 fixed)
N_SYNTH = 200
R, SAMPLE_FRAC = 20, 0.70
rng = np.random.default_rng(0)          # identical draws to val_abstraction_ci.py
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


def jsd(p, q):
    """Jensen-Shannon divergence (base-2, bounded in [0,1]) between two dict-distributions."""
    keys = set(p) | set(q)
    P = np.array([p.get(k, 0.0) for k in keys])
    Q = np.array([q.get(k, 0.0) for k in keys])
    P = P / P.sum() if P.sum() > 0 else P
    Q = Q / Q.sum() if Q.sum() > 0 else Q
    M = 0.5 * (P + Q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


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
    print("[load] Paris-581, m=2 signatures + dominant/secondary decomposition")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")

    # MAT-Sum vocabulary at the paper's operating point (m=2, k=5): merge rare signatures
    fs = Counter(sem["sig2"])
    frequent = [s for s, c in fs.items() if c >= TARGET_K and len(s) > 0]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    sem["matsum"] = sem["sig2"].map(lambda s: remap.get(s, s))

    # per-final-state dominant/secondary decomposition (dominant = majority real top1 label
    # among points mapped to that state; secondary = the other element of the signature, if any)
    dom = sem.groupby("matsum")["top1"].agg(lambda s: s.value_counts().index[0]).to_dict()
    def secondary_of(state, dominant):
        others = [a for a in state if a != dominant]
        return others[0] if others else "NONE"
    sec = {s: secondary_of(s, dom[s]) for s in dom}
    sem["dominant"] = sem["matsum"].map(dom)
    sem["secondary"] = sem["matsum"].map(sec)

    tids = sem["tid"].unique()
    seqs_by_state = {}
    dominant_by_tid = {}
    secondary_by_tid = {}
    for tid, g in sem.groupby("tid", sort=False):
        seqs_by_state[tid] = rle(g["matsum"].tolist())
        # per-visit (post RLE) dominant/secondary, aligned with the RLE-collapsed state sequence
        states_rle, dominants_rle, secondaries_rle = [], [], []
        for st, d, sc in zip(g["matsum"], g["dominant"], g["secondary"]):
            if not states_rle or states_rle[-1] != st:
                states_rle.append(st); dominants_rle.append(d); secondaries_rle.append(sc)
        dominant_by_tid[tid] = dominants_rle
        secondary_by_tid[tid] = secondaries_rle

    n_users = int(len(tids) * SAMPLE_FRAC)
    print(f"[repeat] R={R} paired samples of {n_users}/{len(tids)} users (identical draws to val_abstraction_ci.py)")
    baseline_jsds, synth_jsds = [], []
    for r in range(R):
        sample = rng.choice(tids, n_users, replace=False)
        grng = np.random.default_rng(100 + r)

        # ---- REAL mixed-use profile: P(secondary | dominant=c), weighted by real category freq
        real_dom, real_sec = [], []
        for t in sample:
            real_dom.extend(dominant_by_tid[t]); real_sec.extend(secondary_by_tid[t])
        cat_count = Counter(real_dom)
        total = sum(cat_count.values())
        profile_real = defaultdict(Counter)
        for d, s in zip(real_dom, real_sec):
            profile_real[d][s] += 1
        marginal_sec = Counter(real_sec)

        # baseline: category-specific real profile vs context-free real marginal
        b = sum((cat_count[c] / total) * jsd(profile_real[c], marginal_sec) for c in cat_count)
        baseline_jsds.append(b)

        # ---- MAT-Sum-Markov synthetic: generate, decompose each synthetic state, same profile
        real_seqs = [seqs_by_state[t] for t in sample]
        synth = markov_gen(real_seqs, N_SYNTH, grng)
        profile_synth = defaultdict(Counter)
        for seq in synth:
            for st in seq:
                d = dom.get(st); s = sec.get(st)
                if d is not None:
                    profile_synth[d][s] += 1
        s_div = sum((cat_count[c] / total) * jsd(profile_real[c], profile_synth[c]) for c in cat_count)
        synth_jsds.append(s_div)
        print(f"  rep {r + 1}/{R}: baseline_JSD={b:.4f}  synth_JSD={s_div:.4f}", flush=True)

    def summarize(a, name):
        a = np.array(a, float)
        ci = TCRIT * a.std(ddof=1) / np.sqrt(len(a))
        print(f"{name}: mean={a.mean():.4f} ± {ci:.4f} (95% CI), median={np.median(a):.4f}")
        return a

    ba = summarize(baseline_jsds, "baseline_JSD (Top1 loses this: category-specific vs context-free real profile)")
    sa = summarize(synth_jsds, "synth_JSD    (MAT-Sum release: real vs synthetic category-specific profile)")

    d = ba - sa
    from scipy.stats import wilcoxon, shapiro
    dz = d.mean() / d.std(ddof=1)
    brng = np.random.default_rng(7)
    boot = np.array([brng.choice(d, len(d), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wp = float(wilcoxon(d).pvalue)
    sp = float(shapiro(d).pvalue)
    print(f"\n[paired] baseline_JSD - synth_JSD: mean={d.mean():+.4f} (median={np.median(d):+.4f})  "
          f"CI95=[{lo:+.4f},{hi:+.4f}]  d_z={dz:.1f}  p_wilcox={wp:.1e} (Shapiro={sp:.2f})")

    res_dir = os.path.join(OUT.replace("/data/paris", "/results/paris"))
    os.makedirs(res_dir, exist_ok=True)
    pd.DataFrame({"rep": range(R), "baseline_JSD": baseline_jsds, "synth_JSD": synth_jsds}).to_csv(
        os.path.join(res_dir, "val_mixeduse_downstream.csv"), index=False)
    print("saved:", os.path.join(res_dir, "val_mixeduse_downstream.csv"))

    # a few example categories, pooled over the full population, for a human-readable illustration
    print("\n[illustration] pooled over all 581 users, top dominant categories by frequency:")
    all_dom, all_sec = dominant_by_tid, secondary_by_tid
    pooled_dom, pooled_sec = [], []
    for t in tids:
        pooled_dom.extend(all_dom[t]); pooled_sec.extend(all_sec[t])
    cat_count_full = Counter(pooled_dom)
    profile_full = defaultdict(Counter)
    for d_, s_ in zip(pooled_dom, pooled_sec):
        profile_full[d_][s_] += 1
    for c, n in cat_count_full.most_common(6):
        top_secs = profile_full[c].most_common(3)
        frac_none = profile_full[c]["NONE"] / n
        print(f"  {c:20s} (n={n:6d}, {(1-frac_none)*100:4.1f}% mixed-use) -> top secondary aspects: {top_secs}")


if __name__ == "__main__":
    main()
