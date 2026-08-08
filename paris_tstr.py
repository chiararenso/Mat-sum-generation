"""
TSTR sketch on Paris-581 -- a first utility (downstream-task) evaluation, the missing leg of the triad.
Task: next-OSM-label prediction in the fixed common label space (given the current semantic label,
predict the next one). A fixed 20% of individuals is held out as a REAL test set, never used for
training. For each training size n (20 nested paired replicates from the remaining pool) we:
  - build the MAT-Sum vocabulary on the n training individuals and generate 200 synthetic sequences;
  - TSTR: fit the predictor on the SYNTHETIC label sequences, evaluate top-1 accuracy on the real test set;
  - TRTR: fit on the n REAL training label sequences (upper bound); evaluate on the same test set;
  - base: majority next-label from training (lower bound).
Utility retained = (TSTR - base) / (TRTR - base). Order-1 and order-2 predictors (the order-2 row shows
how much higher-order utility the order-1 Markov generator's synthetic data still supports).
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS
from val_rq3 import rle, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K, N_SYN, R, TEST_FRAC = 2, 5, 200, 20, 0.20
NS = [20, 50, 100, 300, 465]
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093


def fit1(seqs):
    trans = defaultdict(Counter); marg = Counter()
    for v in seqs:
        for a, b in zip(v, v[1:]):
            trans[a][b] += 1; marg[b] += 1
    pred = {a: c.most_common(1)[0][0] for a, c in trans.items()}
    glob = marg.most_common(1)[0][0] if marg else None
    return pred, glob


def fit2(seqs):
    trans = defaultdict(Counter)
    for v in seqs:
        for a, b, c in zip(v, v[1:], v[2:]):
            trans[(a, b)][c] += 1
    pred = {k: c.most_common(1)[0][0] for k, c in trans.items()}
    return pred


def acc1(pred, glob, test):
    ok = tot = 0
    for v in test:
        for a, b in zip(v, v[1:]):
            ok += (pred.get(a, glob) == b); tot += 1
    return ok / tot if tot else 0.0


def acc2(pred2, pred1, glob, test):
    ok = tot = 0
    for v in test:
        for a, b, c in zip(v, v[1:], v[2:]):
            p = pred2.get((a, b)) or pred1.get(b, glob)     # order-2 -> backoff order-1 -> marginal
            ok += (p == c); tot += 1
    return ok / tot if tot else 0.0


def dom_and_labels(sample_tids, per):
    raw = {t: rle(per[t][0]) for t in sample_tids}
    fs = Counter(s for t in sample_tids for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in sample_tids]
    dom = defaultdict(Counter)
    for t in sample_tids:
        for s, l in zip(per[t][0], per[t][1]):
            dom[remap[s]][l] += 1
    dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}
    real_lab = [rle([per[t][1][i] for i in range(len(per[t][1]))]) for t in sample_tids]  # real top1 label seq
    return real, dom, real_lab


def main():
    print("[load] Paris-581", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: (list(g["sig2"]), list(g["top1"])) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))

    # fixed held-out real test set
    rng0 = np.random.default_rng(2024)
    perm0 = rng0.permutation(tids); n_test = int(TEST_FRAC * len(tids))
    test_tids = perm0[:n_test]; pool = perm0[n_test:]
    test_lab = [rle(per[t][1]) for t in test_tids]
    print(f"  test individuals={len(test_tids)}  train pool={len(pool)}", flush=True)

    rows = []
    for r in range(R):
        perm = np.random.default_rng(1000 + r).permutation(pool)
        for n in NS:
            sample = list(perm[:n])
            real, dom, real_lab = dom_and_labels(sample, per)
            grng = np.random.default_rng(40_000 * r + n)
            synth = markov(real, N_SYN, grng)
            synth_lab = [rle([dom[s] for s in v]) for v in synth]
            # order-1
            p1_s, g_s = fit1(synth_lab); p1_r, g_r = fit1(real_lab)
            base = acc1({}, g_r, test_lab)
            tstr1 = acc1(p1_s, g_s, test_lab); trtr1 = acc1(p1_r, g_r, test_lab)
            # order-2
            p2_s = fit2(synth_lab); p2_r = fit2(real_lab)
            tstr2 = acc2(p2_s, p1_s, g_s, test_lab); trtr2 = acc2(p2_r, p1_r, g_r, test_lab)
            rows.append({"rep": r, "n": n, "base": base, "TSTR1": tstr1, "TRTR1": trtr1,
                         "TSTR2": tstr2, "TRTR2": trtr2})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "paris_tstr_raw.csv"), index=False)

    agg = []
    for n in NS:
        sub = raw[raw.n == n]; row = {"n": n}
        for m in ["base", "TSTR1", "TRTR1", "TSTR2", "TRTR2"]:
            a = sub[m].to_numpy(float)
            row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        ret1 = (sub.TSTR1 - sub.base) / (sub.TRTR1 - sub.base)
        ret2 = (sub.TSTR2 - sub.base) / (sub.TRTR2 - sub.base)
        row["retained1"] = round(float(ret1.mean()), 3); row["retained1_ci"] = round(float(TCRIT * ret1.std(ddof=1) / np.sqrt(len(ret1))), 3)
        row["retained2"] = round(float(ret2.mean()), 3); row["retained2_ci"] = round(float(TCRIT * ret2.std(ddof=1) / np.sqrt(len(ret2))), 3)
        agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "paris_tstr_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== Paris TSTR (next-label top-1 accuracy, real test set) ===")
    print(adf[["n", "base", "TRTR1", "TSTR1", "retained1", "TRTR2", "TSTR2", "retained2"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.array(NS); fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].errorbar(x, adf.TRTR1, yerr=adf.TRTR1_ci, fmt="o-", color="#2c7fb8", capsize=3, lw=2, label="TRTR (train real)")
    ax[0].errorbar(x, adf.TSTR1, yerr=adf.TSTR1_ci, fmt="s-", color="#d95f0e", capsize=3, lw=2, label="TSTR (train synth)")
    ax[0].plot(x, adf.base, "k:", lw=1.5, label="majority baseline")
    ax[0].set_title("(A) Next-label accuracy (order-1)"); ax[0].set_ylabel("top-1 accuracy on real test (↑)")
    ax[1].errorbar(x, adf.retained1, yerr=adf.retained1_ci, fmt="o-", color="#6a51a3", capsize=3, lw=2, label="order-1")
    ax[1].errorbar(x, adf.retained2, yerr=adf.retained2_ci, fmt="s--", color="#2f8f4f", capsize=3, lw=2, label="order-2")
    ax[1].axhline(1.0, color="#888", ls=":", lw=1); ax[1].set_ylim(0, 1.15)
    ax[1].set_title("(B) Utility retained  (TSTR−base)/(TRTR−base)"); ax[1].set_ylabel("fraction retained (↑)"); ax[1].legend(fontsize=9)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off()
    ax[0].legend(fontsize=9)
    fig.suptitle("Paris-581 · TSTR utility · next-semantic-label prediction · 20 replicates (95% CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIG, "paris_tstr.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
