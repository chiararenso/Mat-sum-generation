"""
Membership Inference Attack on Paris-581 (the missing MI leg of the privacy triad).
Standard distance-to-closest-record attack against synthetic data: an attacker holding a candidate
real record x scores it by its maximum MUITAS similarity to the released synthetic set, and guesses
"member" (in the generator's training set) when the score is high. We measure how well this separates
true members (the n training individuals) from non-members (a fixed held-out set never used in training).
  attack score  s(x) = max_j MUITAS(x, synth_j)
  metrics       ROC-AUC (0.5 = no membership signal, i.e. safe) and TPR at FPR = 10% (tail of the attack)
Reported across n over 20 replicates. A well-generalising generator gives AUC ~ 0.5.
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
import matsum_summarize as MS
from val_rq3 import rle, subset_matrix, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K, N_SYN, R, TEST_FRAC = 2, 5, 400, 20, 0.20
NS = [20, 50, 100, 300, 465]
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093


def tpr_at_fpr(y, s, fpr_target=0.10):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def map_ids(raw_sig2, freq_set, freq_list, sid):
    seq = [(s if s in freq_set else max(freq_list, key=lambda f: jac(s, f))) for s in rle(raw_sig2)]
    return np.array([sid[s] for s in rle(seq)])


def main():
    print("[load] Paris-581", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: list(g["sig2"]) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))
    rng0 = np.random.default_rng(2024); perm0 = rng0.permutation(tids)
    n_test = int(TEST_FRAC * len(tids)); nonmember = perm0[:n_test]; pool = perm0[n_test:]
    print(f"  non-members(held-out)={len(nonmember)}  train pool={len(pool)}", flush=True)

    rows = []
    for r in range(R):
        perm = np.random.default_rng(1000 + r).permutation(pool)
        for n in NS:
            sample = list(perm[:n])
            raw = {t: rle(per[t]) for t in sample}
            fs = Counter(s for t in sample for s in raw[t])
            freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
            freq_set = set(freq)
            remap = {s: (s if s in freq_set else max(freq, key=lambda f: jac(s, f))) for s in fs}
            real = [rle([remap[s] for s in raw[t]]) for t in sample]
            symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
            sid = {s: i for i, s in enumerate(symbols)}; Rm = subset_matrix(symbols)
            member_ids = [np.array([sid[s] for s in v]) for v in real]
            nonmember_ids = [map_ids(per[t], freq_set, freq, sid) for t in nonmember]
            grng = np.random.default_rng(50_000 * r + n)
            synth = markov(real, N_SYN, grng)
            synth_ids = [np.array([sid[s] for s in v]) for v in synth]
            sm = [max((sim(Rm, x, z) for z in synth_ids), default=0.0) for x in member_ids]
            sn = [max((sim(Rm, x, z) for z in synth_ids), default=0.0) for x in nonmember_ids]
            y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]; s = np.r_[sm, sn]
            auc = roc_auc_score(y, s); tpr10 = tpr_at_fpr(y, s, 0.10)
            rows.append({"rep": r, "n": n, "auc": auc, "tpr@10": tpr10,
                         "mem_mean": float(np.mean(sm)), "non_mean": float(np.mean(sn))})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw_df = pd.DataFrame(rows); raw_df.to_csv(os.path.join(OUT, "paris_mia_raw.csv"), index=False)

    agg = []
    for n in NS:
        sub = raw_df[raw_df.n == n]; row = {"n": n}
        for m in ["auc", "tpr@10", "mem_mean", "non_mean"]:
            a = sub[m].to_numpy(float)
            row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "paris_mia_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== Paris MIA (distance-to-closest-synthetic) ===")
    print(adf[["n", "auc", "auc_ci", "tpr@10", "tpr@10_ci", "mem_mean", "non_mean"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.array(NS); fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].errorbar(x, adf.auc, yerr=adf.auc_ci, fmt="o-", color="#8856a7", capsize=3, lw=2)
    ax[0].axhline(0.5, color="#888", ls="--", lw=1.2, label="random (AUC=0.5, safe)")
    ax[0].set_ylim(0.40, 0.75); ax[0].set_title("(A) Membership AUC"); ax[0].set_ylabel("ROC-AUC (0.5 = no leakage)"); ax[0].legend(fontsize=9)
    ax[1].errorbar(x, adf["tpr@10"], yerr=adf["tpr@10_ci"], fmt="s-", color="#d95f0e", capsize=3, lw=2)
    ax[1].axhline(0.10, color="#888", ls="--", lw=1.2, label="random (=FPR)")
    ax[1].set_title("(B) TPR at 10% FPR"); ax[1].set_ylabel("true-positive rate (↓ safer)"); ax[1].legend(fontsize=9)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of training individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off()
    fig.suptitle("Paris-581 · membership inference attack · MAT-Sum+Markov · 20 replicates (95% CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIG, "paris_mia.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
