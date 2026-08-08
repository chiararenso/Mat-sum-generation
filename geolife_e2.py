"""
E2 -- scaling small-data on GeoLife (external replication of RQ2), MAT-Sum + order-1 Markov only.
Nested paired user samples D10(r) subset ... subset DU(r); per n rebuild the MAT-Sum vocabulary on the
sampled users, estimate Markov/lengths on the sample, generate 200 synthetic SESSIONS (no cross-session
transitions). Quality = semantic generation TV (fixed OSM-label space). Disclosure risk = exact session
copies/200 and a USER-AWARE DCR: DCR(s)=1-max_u max_{x in T_u} MUITAS(s,x); the real-to-real baseline
compares each real session only to sessions of OTHER users and averages within-user then across-users.
Also: representation diagnostics + an estimated saturation point (segmented fit of TV vs log n, bootstrap).
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
from geolife_e1 import load, rle, bigram, tv, submat, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, N_SYN, R, REF = 5, 200, 20, 400
NS = [10, 20, 30, 50, 75, 100, 150, 152]
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093


def run(sess, users, grng):
    # sess: dict tid -> (sig2_list, top1_list, user)
    tids = [t for t in sess if sess[t][2] in users]
    raw = {t: rle(sess[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids]
    real_user = [sess[t][2] for t in tids]
    dom = defaultdict(Counter); gt = []
    for t in tids:
        sig_l, top_l, _ = sess[t]
        for s, l in zip(sig_l, top_l):
            dom[remap[s]][l] += 1
        gt.append(rle(top_l))
    dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = submat(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]
    gt_bi = bigram(gt)

    synth = markov(real, N_SYN, grng)
    synth_ids = [np.array([sid[s] for s in v]) for v in synth]
    synth_lab = [rle([dom[s] for s in v]) for v in synth]
    sem_gen = tv(bigram(synth_lab), gt_bi)
    real_tup = set(tuple(v) for v in real)
    exact = sum(tuple(v) in real_tup for v in synth)

    # ---- reference (cap) with user labels ----
    ridx = grng.choice(len(real_ids), min(REF, len(real_ids)), replace=False)
    ref_ids = [real_ids[i] for i in ridx]; ref_user = [real_user[i] for i in ridx]
    # DCR synth->training (user-aware max == max over all sessions)
    dcr_s = float(np.mean([1 - max((sim(Rm, s, r) for r in ref_ids), default=0.0) for s in synth_ids]))
    # DCR real->real baseline: each query vs sessions of OTHER users; avg within-user then across-user
    qidx = grng.choice(len(real_ids), min(REF, len(real_ids)), replace=False)
    per_user = defaultdict(list)
    for qi in qidx:
        u = real_user[qi]
        best = max((sim(Rm, real_ids[qi], ref_ids[j]) for j in range(len(ref_ids)) if ref_user[j] != u), default=0.0)
        per_user[u].append(1 - best)
    dcr_b = float(np.mean([np.mean(v) for v in per_user.values()])) if per_user else 0.0

    supp = Counter(s for v in real for s in v)
    sv = np.array(list(supp.values()))
    obs_bi = set().union(*[{(a, b) for a, b in zip(v, v[1:])} for v in real]) if real else set()
    diag = {"n_states": len(symbols), "obs_bigrams": len(obs_bi),
            "median_support": int(np.median(sv)) if len(sv) else 0,
            "sessions": len(tids), "visits": int(sum(len(v) for v in real)),
            "transitions": int(sum(max(0, len(v) - 1) for v in real)),
            "pct_supp_lt5": round(100 * float((sv < 5).mean()), 1) if len(sv) else 0.0,
            "pct_supp_lt10": round(100 * float((sv < 10).mean()), 1) if len(sv) else 0.0}
    return {"sem_genTV": sem_gen, "exact_copies": exact, "DCR_synth": dcr_s, "DCR_base": dcr_b,
            "dDCR": dcr_s - dcr_b, **diag}


def saturation(raw_df):
    """2-segment fit of sem_genTV vs log n; breakpoint by min SSE; bootstrap over replicates."""
    ns = np.array(NS, float); logn = np.log(ns)
    def bp(mean_tv):
        best, bx = 1e18, ns[1]
        for k in range(1, len(ns) - 1):
            e = 0.0
            for sl in (slice(0, k + 1), slice(k, len(ns))):
                X = np.vstack([logn[sl], np.ones(logn[sl].size)]).T; y = mean_tv[sl]
                c, *_ = np.linalg.lstsq(X, y, rcond=None); e += ((X @ c - y) ** 2).sum()
            if e < best:
                best, bx = e, ns[k]
        return bx
    reps = sorted(raw_df.rep.unique()); brng = np.random.default_rng(3); bps = []
    for _ in range(2000):
        samp = brng.choice(reps, len(reps), replace=True)
        mt = np.array([raw_df[(raw_df.n == n) & (raw_df.rep.isin(samp))].sem_genTV.mean() for n in NS])
        bps.append(bp(mt))
    mean_tv = np.array([raw_df[raw_df.n == n].sem_genTV.mean() for n in NS])
    return bp(mean_tv), np.percentile(bps, [2.5, 97.5])


def main():
    sem = load()
    sess = {t: (g["sig2"].tolist(), g["top1"].tolist(), g["user"].iloc[0])
            for t, g in sem.groupby("tid", sort=False)}
    users = np.array(sorted(sem.user.unique())); U = len(users)
    ns = [n for n in NS if n <= U]
    print(f"[E2] U={U}, n={ns}, {R} nested paired replicates")
    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(users)
        for n in ns:
            res = run(sess, set(perm[:n]), np.random.default_rng(10_000 * r + n))
            rows.append({"rep": r, "n": n, **res})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "geolife_e2_raw.csv"), index=False)

    mets = ["sem_genTV", "exact_copies", "DCR_synth", "DCR_base", "dDCR", "n_states", "obs_bigrams",
            "median_support", "sessions", "visits", "transitions", "pct_supp_lt5", "pct_supp_lt10"]
    agg = []
    for n in ns:
        sub = raw[raw.n == n]; row = {"n": n}
        for m in mets:
            a = sub[m].to_numpy(float)
            row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "geolife_e2_summary.csv"), index=False)
    sat, sat_ci = saturation(raw)
    print("\n=== E2 summary (mean +/- 95% CI) ===")
    print(adf[["n", "sem_genTV", "exact_copies", "DCR_synth", "DCR_base", "dDCR", "n_states", "obs_bigrams", "median_support"]].to_string(index=False))
    print(f"\nestimated saturation point (segmented fit, bootstrap): n* = {sat:.0f}  95% CI [{sat_ci[0]:.0f}, {sat_ci[1]:.0f}]")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.array(ns); fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    ax[0].errorbar(x, adf.sem_genTV, yerr=adf.sem_genTV_ci, fmt="o-", color="#2c7fb8", capsize=3, lw=2)
    ax[0].axvline(sat, color="#888", ls=":", label=f"n*≈{sat:.0f}"); ax[0].legend(fontsize=9)
    ax[0].set_title("(A) Fidelity"); ax[0].set_ylabel("semantic generation TV (↓)")
    ax[1].errorbar(x, adf.DCR_synth, yerr=adf.DCR_synth_ci, fmt="s-", color="#d95f0e", capsize=3, lw=2, label="synth→training")
    ax[1].errorbar(x, adf.DCR_base, yerr=adf.DCR_base_ci, fmt="o--", color="#555", capsize=3, lw=2, label="real→real (user-aware)")
    ax[1].set_title("(B) Distance to real records"); ax[1].set_ylabel("DCR (↑ safer)"); ax[1].legend(fontsize=9)
    ax[2].errorbar(x, adf.exact_copies, yerr=adf.exact_copies_ci, fmt="D-", color="#6a53a6", capsize=3, lw=2)
    ax[2].set_title("(C) Exact session copies"); ax[2].set_ylabel("exact copies (of 200)"); ax[2].axhline(0, color="#ccc", lw=.8)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off()
    fig.suptitle(f"E2 GeoLife · MAT-Sum+Markov scaling · {R} nested paired replicates · fixed abstraction parameters", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "geolife_e2.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("figure:", p)


if __name__ == "__main__":
    main()
