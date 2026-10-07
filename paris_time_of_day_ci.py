"""
Time-of-day activity demand (Sec. 5.3), revised protocol: Paris-OSM, MAT-Sum m=2,k=5, order-1 Markov.

Differences from paris_time_of_day.py (which reproduces the originally published numbers):
  * hours are LOCAL Paris time (Europe/Paris), not UTC;
  * R=20 independent generations of N=200 synthetic trajectories -> mean and 95% CI;
  * the headline numbers are WITHIN-category hourly profiles (per-category JSD and Pearson r, and their
    visit-weighted mean). The flattened category x hour r of the original analysis is also reported, but it is
    dominated by differences between category sizes, not by timing;
  * a real-vs-real floor: the same measures between two disjoint random halves of the real individuals
    (each category profile normalised separately), per replicate.
Real = visits of the MAT-Sum state projection (run-length collapse of the remapped signature) labelled with the
dominant OSM label of their state. Synthetic start times are bootstrapped from the real first-visit local times
and subsequent visit times accumulate the sampled dwell times (no gaps), as in the original protocol.
"""
import os
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import matsum_summarize as MS
from matsum_synth import build_visits, anonymize_vocabulary, SemanticMarkov
import matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", ".")
TZ = "Europe/Paris"
M_TOP, K_SUPPORT, N_SYNTH, R, TOP = 2, 5, 200, 20, 7
TCRIT = 2.093  # t_{0.975, 19}


def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def measures(A, B, w):
    """A, B: (TOP, 24) count matrices. Per-category JSD/r, their weighted means, and the flattened JSD/r."""
    js = np.array([jsd(A[i] + 1e-12, B[i] + 1e-12) for i in range(TOP)])
    rr = np.array([pearson(A[i], B[i]) for i in range(TOP)])
    ok = ~np.isnan(rr)
    return js, rr, float(np.average(js, weights=w)), float(np.average(rr[ok], weights=w[ok])), \
        jsd(A.ravel() + 1e-12, B.ravel() + 1e-12), pearson(A.ravel(), B.ravel())


def main():
    print(f"[setup] semantic mapping + vocabulary (m={M_TOP}, k={K_SUPPORT})", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    raw = build_visits(sem, M_TOP)
    seqs, _, _, _ = anonymize_vocabulary(raw, K_SUPPORT)
    fs = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, c in fs.items() if c >= K_SUPPORT and len(s) > 0] or [fs.most_common(1)[0][0]]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    sig2 = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:M_TOP]) if hasattr(x, "__len__") else frozenset())
    votes = defaultdict(Counter)
    for sig, t1 in zip(sig2, sem["top1"]):
        votes[remap.get(sig, sig)][t1] += 1
    dom = {s: c.most_common(1)[0][0] for s, c in votes.items()}
    cats = [c for c, _ in Counter(dom.values()).most_common(TOP)]
    cidx = {c: i for i, c in enumerate(cats)}

    # ---- REAL: per-individual (category x local hour) counts + first-visit local hour ----
    loc = sem["time"].dt.tz_convert(TZ)
    sem_state = [remap.get(s, s) for s in sig2]
    per_ind, starts, tids = [], [], []
    for tid, idx in sem.groupby("tid", sort=False).indices.items():
        M = np.zeros((TOP, 24)); prev = None; first = None
        for i in idx:
            st = sem_state[i]
            if st != prev:
                c = cidx.get(dom.get(st, "other"))
                if c is not None:
                    M[c, loc.iloc[i].hour] += 1
                if first is None:
                    first = loc.iloc[i].hour + loc.iloc[i].minute / 60.0
                prev = st
        per_ind.append(M); starts.append(first); tids.append(tid)
    per_ind = np.array(per_ind); starts = np.array(starts, float)
    RC = per_ind.sum(0); w = RC.sum(1) / RC.sum()
    print(f"  top categories: {cats}\n  real: {len(tids)} individuals, {int(RC.sum())} categorized visits", flush=True)

    model = SemanticMarkov(alpha=0.1).fit(seqs)
    rows, prof_all = [], []
    for r in range(R):
        SY.rng = np.random.default_rng(100 + r)
        grng = np.random.default_rng(500 + r)
        SC = np.zeros((TOP, 24))
        for _ in range(N_SYNTH):
            v = model.sample(200)
            if not v:
                continue
            clock = float(grng.choice(starts))
            for sym, dwell_s in v:
                c = cidx.get(dom.get(sym, "other"))
                if c is not None:
                    SC[c, int(clock) % 24] += 1
                clock += dwell_s / 3600.0
        prof_all.append(SC)
        js, rr, jw, rw, jf, rf = measures(RC, SC, w)
        perm = np.random.default_rng(900 + r).permutation(len(tids)); h = len(tids) // 2
        A = per_ind[perm[:h]].sum(0); B = per_ind[perm[h:2 * h]].sum(0)
        fjs, frr, fjw, frw, fjf, frf = measures(A, B, w)
        for i, c in enumerate(cats):
            rows.append({"rep": r, "level": "category", "category": c, "synth_JSD": js[i], "synth_r": rr[i],
                         "floor_JSD": fjs[i], "floor_r": frr[i]})
        rows.append({"rep": r, "level": "weighted", "category": "visit-weighted mean", "synth_JSD": jw, "synth_r": rw,
                     "floor_JSD": fjw, "floor_r": frw})
        rows.append({"rep": r, "level": "flattened", "category": "flattened (category x hour)", "synth_JSD": jf,
                     "synth_r": rf, "floor_JSD": fjf, "floor_r": frf})
        print(f"  replicate {r + 1}/{R} done", flush=True)
    raw_df = pd.DataFrame(rows); raw_df.to_csv(os.path.join(OUT, "paris_time_of_day_ci_raw.csv"), index=False)

    agg = []
    for (lv, c), g in raw_df.groupby(["level", "category"], sort=False):
        row = {"level": lv, "category": c}
        for m in ["synth_JSD", "synth_r", "floor_JSD", "floor_r"]:
            a = g[m].to_numpy(float); a = a[~np.isnan(a)]
            row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        agg.append(row)
    adf = pd.DataFrame(agg)
    adf["real_share_pct"] = [round(100 * w[cidx[c]], 1) if c in cidx else np.nan for c in adf.category]
    adf.to_csv(os.path.join(OUT, "paris_time_of_day_ci_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== Time-of-day demand, local Paris hours, R={R} generations of N={N_SYNTH} (mean ± 95% CI) ===")
    t = adf.copy()
    for m in ["synth_JSD", "synth_r", "floor_JSD", "floor_r"]:
        t[m] = t[m].map("{:.3f}".format) + "±" + t[m + "_ci"].map("{:.3f}".format)
    print(t[["category", "real_share_pct", "synth_JSD", "floor_JSD", "synth_r", "floor_r"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    S = np.array(prof_all); hrs = np.arange(24)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharex=True); ax = ax.flatten()
    for i, c in enumerate(cats):
        a = ax[i]; real = 100 * RC[i] / RC[i].sum()
        sy = 100 * S[:, i, :] / S[:, i, :].sum(1, keepdims=True)
        m, ci = sy.mean(0), TCRIT * sy.std(0, ddof=1) / np.sqrt(R)
        a.plot(hrs, real, "-o", ms=3, lw=1.8, color="#2c7fb8", label="real")
        a.plot(hrs, m, "-s", ms=3, lw=1.8, color="#d95f0e", label="synthetic (mean, 95% CI)")
        a.fill_between(hrs, m - ci, m + ci, color="#d95f0e", alpha=0.2)
        rowc = adf[(adf.level == "category") & (adf.category == c)].iloc[0]
        a.set_title(f"{c}\nr={rowc.synth_r:.2f}, JSD={rowc.synth_JSD:.3f}", fontsize=10)
        a.set_xticks([0, 6, 12, 18, 23])
        if i == 0:
            a.legend(fontsize=8)
    for i in range(len(cats), 8):
        ax[i].axis("off")
    fig.supxlabel("hour of day (local time, Paris)"); fig.supylabel("% of the category's visits")
    fig.suptitle("Time-of-day activity demand, Paris-OSM: real vs synthetic (within-category profiles)")
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "fig_time_of_day_demand_ci.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("figure:", p)


if __name__ == "__main__":
    main()
