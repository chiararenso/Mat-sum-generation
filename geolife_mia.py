"""
Membership Inference Attack on GeoLife (mirrors paris_mia.py; unit = session, held-out = users).
20% of users are reserved as non-members (their sessions are never in training). For each n over 20
replicates we sample n training users from the remaining pool, generate 300 synthetic sessions, and
score up to 300 sampled member sessions and 300 sampled non-member sessions by their max MUITAS to the
synthetic set. ROC-AUC and TPR@FPR=10%. AUC ~ 0.5 = no membership signal.
"""
import os, pickle, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve
from geolife_e1 import rle, submat, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, N_SYN, R, TEST_FRAC, CAP = 5, 300, 20, 0.20, 300
NS = [20, 50, 100, 121]
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093


def tpr_at_fpr(y, s, f=0.10):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(f, fpr, tpr))


def map_ids(raw_sig2, freq_set, freq_list, sid):
    seq = [(s if s in freq_set else max(freq_list, key=lambda f: jac(s, f))) for s in rle(raw_sig2)]
    return np.array([sid[s] for s in rle(seq)])


def main():
    with open(os.path.join(OUT, "geolife_sess_cache.pkl"), "rb") as f:
        sess, users_all = pickle.load(f)
    users_all = np.array(sorted(users_all))
    rng0 = np.random.default_rng(2024); perm0 = rng0.permutation(users_all)
    n_test = int(TEST_FRAC * len(users_all)); nm_users = set(perm0[:n_test]); pool = perm0[n_test:]
    nm_tids = [t for t in sess if sess[t][2] in nm_users]
    print(f"[geolife MIA] U={len(users_all)} non-member users={len(nm_users)} pool={len(pool)}", flush=True)

    rows = []
    for r in range(R):
        perm = np.random.default_rng(1000 + r).permutation(pool)
        for n in NS:
            train_users = set(perm[:n])
            tids = [t for t in sess if sess[t][2] in train_users]
            raw = {t: rle(sess[t][0]) for t in tids}
            fs = Counter(s for t in tids for s in raw[t])
            freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
            freq_set = set(freq)
            remap = {s: (s if s in freq_set else max(freq, key=lambda f: jac(s, f))) for s in fs}
            real = [rle([remap[s] for s in raw[t]]) for t in tids]
            symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
            sid = {s: i for i, s in enumerate(symbols)}; Rm = submat(symbols)
            grng = np.random.default_rng(60_000 * r + n)
            # sample members / non-members
            mi = grng.choice(len(tids), min(CAP, len(tids)), replace=False)
            member_ids = [np.array([sid[s] for s in real[i]]) for i in mi]
            ni = grng.choice(len(nm_tids), min(CAP, len(nm_tids)), replace=False)
            nonmember_ids = [map_ids(sess[nm_tids[i]][0], freq_set, freq, sid) for i in ni]
            synth = markov(real, N_SYN, grng)
            synth_ids = [np.array([sid[s] for s in v]) for v in synth]
            sm = [max((sim(Rm, x, z) for z in synth_ids), default=0.0) for x in member_ids]
            sn = [max((sim(Rm, x, z) for z in synth_ids), default=0.0) for x in nonmember_ids]
            y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]; s = np.r_[sm, sn]
            rows.append({"rep": r, "n": n, "auc": roc_auc_score(y, s), "tpr@10": tpr_at_fpr(y, s, 0.10),
                         "mem_mean": float(np.mean(sm)), "non_mean": float(np.mean(sn))})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw_df = pd.DataFrame(rows); raw_df.to_csv(os.path.join(OUT, "geolife_mia_raw.csv"), index=False)

    agg = []
    for n in NS:
        sub = raw_df[raw_df.n == n]; row = {"n": n}
        for m in ["auc", "tpr@10", "mem_mean", "non_mean"]:
            a = sub[m].to_numpy(float)
            row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "geolife_mia_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== GeoLife MIA ===")
    print(adf[["n", "auc", "auc_ci", "tpr@10", "tpr@10_ci", "mem_mean", "non_mean"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.array(NS); fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].errorbar(x, adf.auc, yerr=adf.auc_ci, fmt="o-", color="#2c7fb8", capsize=3, lw=2)
    ax[0].axhline(0.5, color="#888", ls="--", lw=1.2, label="random (AUC=0.5, safe)")
    ax[0].set_ylim(0.40, 0.75); ax[0].set_title("(A) Membership AUC"); ax[0].set_ylabel("ROC-AUC (0.5 = no leakage)"); ax[0].legend(fontsize=9)
    ax[1].errorbar(x, adf["tpr@10"], yerr=adf["tpr@10_ci"], fmt="s-", color="#d95f0e", capsize=3, lw=2)
    ax[1].axhline(0.10, color="#888", ls="--", lw=1.2, label="random (=FPR)")
    ax[1].set_title("(B) TPR at 10% FPR"); ax[1].set_ylabel("true-positive rate (↓ safer)"); ax[1].legend(fontsize=9)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of training individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off()
    fig.suptitle("GeoLife · membership inference attack · MAT-Sum+Markov · 20 replicates (95% CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIG, "geolife_mia.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
