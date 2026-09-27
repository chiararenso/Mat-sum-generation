"""
Release-size vs disclosure risk on GeoLife, mirroring the Paris-OSM release-size analysis
(paper Sec. 5.4 "Release size"). A single order-1 Markov model is trained ONCE on the full
GeoLife matched-size population (all 152 retained users), and we vary the number of RELEASED
synthetic sessions N, recomputing DCR (mean and 5th-percentile tail) and exact copies over a
sampled set of real query sessions against the first N synthetic draws each time (nested:
larger releases contain all smaller ones).

Motivation: GeoLife's matched-size release (Nsynth=Nreal, Table 18) is ~13,370 sessions, far
above the paper's N=200 comparison operating point. This quantifies whether the release-size
risk documented on Paris (worst-case DCR crossing the real-to-real baseline between N=5,000
and N=10,000) also holds at GeoLife's own natural full-release scale, rather than assuming it
transfers from Paris by analogy.
"""
import os, pickle, numpy as np, pandas as pd
from collections import Counter
from geolife_e1 import rle, submat, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K = 5
NS_RELEASE = [200, 1000, 5000, 10000, 13370]
R = 5
N_QUERY = 400          # sampled real query sessions per replicate (tractability; same convention as geolife_disclosure.py)
BASE_OTHER_CAP = 2000  # candidates per query for the real-to-real (leave-one-user-out) baseline
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    with open(os.path.join(OUT, "geolife_sess_cache.pkl"), "rb") as f:
        sess, users_all = pickle.load(f)
    users_all = np.array(sorted(users_all))
    print(f"[geolife release-size] U={len(users_all)} (full matched-size population)", flush=True)

    tids = list(sess.keys())
    raw = {t: rle(sess[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids]
    ru = [sess[t][2] for t in tids]
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = submat(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]
    print(f"  sessions={len(tids)} states={len(symbols)}", flush=True)

    brng = np.random.default_rng(999)
    bidx = brng.choice(len(real_ids), min(N_QUERY, len(real_ids)), replace=False)
    baseline_scores = []
    for q in bidx:
        others = [j for j in range(len(real_ids)) if ru[j] != ru[q]]
        oidx = brng.choice(len(others), min(BASE_OTHER_CAP, len(others)), replace=False)
        cand = [others[i] for i in oidx]
        baseline_scores.append(max((sim(Rm, real_ids[q], real_ids[j]) for j in cand), default=0.0))
    baseline_dcr = 1 - np.array(baseline_scores)
    baseline_mean, baseline_p05 = float(baseline_dcr.mean()), float(np.percentile(baseline_dcr, 5))
    print(f"  real-to-real baseline DCR: mean={baseline_mean:.3f} p05={baseline_p05:.3f}", flush=True)

    N_MAX = max(NS_RELEASE)
    rows = []
    for r in range(R):
        grng = np.random.default_rng(70_000 + r)
        synth = markov(real, N_MAX, grng)                          # one long nested draw
        synth_ids = [np.array([sid[s] for s in v]) for v in synth]
        synth_tuples = [tuple(v) for v in synth]
        qidx = grng.choice(len(real_ids), min(N_QUERY, len(real_ids)), replace=False)

        for qi, q in enumerate(qidx):
            sims = np.array([sim(Rm, real_ids[q], z) for z in synth_ids])   # length N_MAX
            cummax = np.maximum.accumulate(sims)
            is_exact = np.array([real[q] == synth[j] for j in range(N_MAX)])
            first_exact = np.where(is_exact, np.arange(N_MAX), N_MAX)
            cum_first_exact = np.minimum.accumulate(first_exact)
            if qi == 0:
                all_cummax = np.zeros((len(qidx), N_MAX))
                all_first_exact = np.zeros((len(qidx), N_MAX), dtype=int)
            all_cummax[qi] = cummax
            all_first_exact[qi] = cum_first_exact
            if (qi + 1) % 100 == 0:
                print(f"    rep {r+1}: query {qi+1}/{len(qidx)}", flush=True)

        for N in NS_RELEASE:
            sc = all_cummax[:, N - 1]
            dcr = 1 - sc
            exact = int((all_first_exact[:, N - 1] < N).sum())
            rows.append({"rep": r, "N": N,
                         "DCR_mean": float(dcr.mean()), "DCR_p05": float(np.percentile(dcr, 5)),
                         "DCR_min": float(dcr.min()),
                         "exact_copies_among_query": exact, "n_query": len(qidx)})
        print(f"  replicate {r+1}/{R} done", flush=True)

    df = pd.DataFrame(rows)
    df["baseline_DCR_mean"] = baseline_mean
    df["baseline_DCR_p05"] = baseline_p05
    df.to_csv(os.path.join(OUT, "geolife_release_size.csv"), index=False)

    agg = []
    for N in NS_RELEASE:
        sub = df[df.N == N]
        row = {"N": N}
        for m in ["DCR_mean", "DCR_p05", "DCR_min", "exact_copies_among_query"]:
            a = sub[m].to_numpy(float)
            row[m] = round(float(a.mean()), 4)
            row[m + "_std"] = round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0
        agg.append(row)
    adf = pd.DataFrame(agg)
    adf["baseline_DCR_mean"] = baseline_mean
    adf["baseline_DCR_p05"] = baseline_p05
    adf.to_csv(os.path.join(OUT, "geolife_release_size_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== GeoLife release size vs disclosure risk (matched-size population, N up to natural release scale) ===")
    print(adf.to_string(index=False))
    print(f"\nreal-to-real baseline: mean={baseline_mean:.4f} p05={baseline_p05:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].errorbar(adf.N, adf.DCR_mean, yerr=adf.DCR_mean_std, fmt="o-", color="#d95f0e", capsize=3, lw=2, label="mean DCR")
    ax[0].errorbar(adf.N, adf.DCR_p05, yerr=adf.DCR_p05_std, fmt="s--", color="#2c7fb8", capsize=3, lw=2, label="5th-percentile DCR (tail)")
    ax[0].axhline(baseline_mean, color="#555", ls=":", lw=1.5, label="real-to-real baseline (mean)")
    ax[0].axhline(baseline_p05, color="#888", ls=":", lw=1.2, label="real-to-real baseline (p05)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("release size N (log scale)"); ax[0].set_ylabel("DCR (↑ safer)")
    ax[0].set_title("(A) DCR vs release size"); ax[0].legend(fontsize=8)
    ax[1].plot(adf.N, adf.exact_copies_among_query, "^-", color="#555", lw=2)
    ax[1].set_xscale("log"); ax[1].set_xlabel("release size N (log scale)")
    ax[1].set_ylabel(f"exact copies (of {N_QUERY} sampled queries)")
    ax[1].set_title("(B) Exact copies vs release size")
    fig.suptitle("GeoLife · release size vs disclosure risk · matched-size population (152 users) · 5 replicates", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(FIG, "geolife_release_size.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
