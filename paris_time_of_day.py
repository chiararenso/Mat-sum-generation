"""
Time-of-day activity demand profile, real vs. synthetic, Paris-OSM (m=2,k=5).

A classic aggregate-demand utility check: for each hour of the day, what fraction of visits
belong to each activity category (home/work/leisure/...)? Useful for transport/urban-planning
capacity questions ("when is demand for X highest"), complementary to TSTR (per-step
prediction) and the published aggregate-density figure (spatial, not temporal).

The generator (matsum_synth.SemanticMarkov) samples (state, dwell) sequences with NO absolute
clock anchor -- there is no notion of what time of day a synthetic visit happens at. We add one
the same way the paper already samples trajectory LENGTH empirically (DITRAS-style bootstrap):
each synthetic trajectory's START time is drawn from the empirical distribution of REAL
trajectory start times, then subsequent visit clock-times are obtained by accumulating the
already-sampled dwell durations. This is a minimal, non-spatial addition -- it does not touch
the fragile tile-grounding machinery that caused the OD-matrix check to underperform.
"""
import os
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import matsum_summarize as MS
from matsum_synth import build_visits, anonymize_vocabulary, SemanticMarkov
import matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", ".")
M_TOP, K_SUPPORT, N_SYNTH = 2, 5, 200
N_HOURS = 24
TOP_CATEGORIES = 7


def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    print(f"[setup] semantic mapping + vocabulary (m={M_TOP}, k={K_SUPPORT})", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem = sem.sort_values(["tid", "time"]).reset_index(drop=True)

    raw = build_visits(sem, M_TOP)
    seqs, support, _, _ = anonymize_vocabulary(raw, K_SUPPORT)
    fs = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, c in fs.items() if c >= K_SUPPORT and len(s) > 0] or [fs.most_common(1)[0][0]]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}

    # dominant top1 label per (remapped) state -- for readable category names
    dom_votes = defaultdict(Counter)
    for tid, g in sem.groupby("tid", sort=False):
        pass
    sig2 = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:M_TOP]) if hasattr(x, "__len__") else frozenset())
    for sig, top1 in zip(sig2, sem["top1"]):
        dom_votes[remap.get(sig, sig)][top1] += 1
    dom = {s: c.most_common(1)[0][0] for s, c in dom_votes.items()}

    top_cats = [c for c, _ in Counter(dom.values()).most_common(TOP_CATEGORIES)]
    print(f"  top categories: {top_cats}", flush=True)

    # ---- REAL: (category, hour) matrix directly from real timestamps ----
    real_runs = defaultdict(list)  # tid -> [(state, start_time, dwell_s)]
    for tid, g in sem.groupby("tid", sort=False):
        g = g.reset_index(drop=True)
        prev_sig = None; run_start = None
        sigs = [frozenset(list(x)[:M_TOP]) if hasattr(x, "__len__") else frozenset() for x in g["label_tfidf"]]
        times = g["time"].tolist()
        for i, sig in enumerate(sigs):
            st = remap.get(sig, sig)
            if st != prev_sig:
                if prev_sig is not None:
                    dwell = (times[i] - run_start).total_seconds()
                    real_runs[tid].append((prev_sig, run_start, dwell))
                run_start = times[i]; prev_sig = st
        if prev_sig is not None:
            real_runs[tid].append((prev_sig, run_start, 0.0))

    real_starts = []  # trajectory start times (hour-of-day + fractional), for the synthetic bootstrap
    RC = np.zeros((TOP_CATEGORIES, N_HOURS))
    for tid, runs in real_runs.items():
        if runs:
            t0 = runs[0][1]
            real_starts.append(t0.hour + t0.minute / 60.0)
        for st, t, dwell in runs:
            cat = dom.get(st, "other")
            if cat in top_cats:
                RC[top_cats.index(cat), t.hour] += 1
    real_starts = np.array(real_starts)
    print(f"  real: {len(real_runs)} individuals, {int(RC.sum())} categorized visits", flush=True)

    # ---- SYNTHETIC: sample sequences, bootstrap a start time, accumulate dwell ----
    model = SemanticMarkov(alpha=0.1).fit(seqs)
    SY.rng = np.random.default_rng(0)
    synth = [model.sample(200) for _ in range(N_SYNTH)]
    grng = np.random.default_rng(1)

    SC = np.zeros((TOP_CATEGORIES, N_HOURS))
    for v in synth:
        if not v:
            continue
        start_hour = float(grng.choice(real_starts))
        clock = start_hour  # hours, wraps mod 24
        for sym, dwell_s in v:
            cat = dom.get(sym, "other")
            if cat in top_cats:
                SC[top_cats.index(cat), int(clock) % 24] += 1
            clock += dwell_s / 3600.0

    print(f"  synthetic: {N_SYNTH} trajectories, {int(SC.sum())} categorized visits", flush=True)

    j = jsd(RC.flatten() + 1e-12, SC.flatten() + 1e-12)
    r = float(np.corrcoef(RC.flatten(), SC.flatten())[0, 1])
    print(f"\n=== Time-of-day demand fidelity (category x hour, top {TOP_CATEGORIES} categories) ===")
    print(f"  JSD={j:.4f}  Pearson r={r:.4f}")

    RCn = RC / RC.sum(); SCn = SC / SC.sum()
    rows = []
    for i, cat in enumerate(top_cats):
        cj = jsd(RC[i] + 1e-12, SC[i] + 1e-12)
        cr = float(np.corrcoef(RC[i], SC[i])[0, 1]) if RC[i].std() > 0 and SC[i].std() > 0 else float("nan")
        rows.append({"category": cat, "JSD": round(cj, 4), "pearson_r": round(cr, 4),
                     "real_share_pct": round(100 * RC[i].sum() / RC.sum(), 1),
                     "synth_share_pct": round(100 * SC[i].sum() / SC.sum(), 1)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT, "time_of_day_demand_summary.csv"), index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    ax = ax.flatten()
    hrs = np.arange(N_HOURS)
    for i, cat in enumerate(top_cats):
        a = ax[i]
        a.plot(hrs, RCn[i] * 100, "-o", ms=3, lw=1.8, color="#2c7fb8", label="real")
        a.plot(hrs, SCn[i] * 100, "-s", ms=3, lw=1.8, color="#d95f0e", label="synthetic")
        a.set_title(cat, fontsize=10)
        a.set_xticks([0, 6, 12, 18, 23])
        if i == 0:
            a.legend(fontsize=8)
    for i in range(len(top_cats), 8):
        ax[i].axis("off")
    fig.supxlabel("hour of day"); fig.supylabel("% of all categorized visits")
    fig.suptitle(f"Time-of-day activity demand, Paris-OSM, real vs. synthetic (overall JSD={j:.3f}, r={r:.3f})")
    fig.tight_layout()
    p = os.path.join(FIG, "fig_time_of_day_demand.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig)
    print("\nfigure:", p)


if __name__ == "__main__":
    main()
