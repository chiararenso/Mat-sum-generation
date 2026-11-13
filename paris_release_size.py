"""
Release-size vs. disclosure risk, Paris-581 (public). RECOVERED from the session in which it was run (it had not been committed)., MAT-Sum + order-1 Markov, m=2,k=5.

Trains the generator ONCE on the full 581 individuals (a single fixed model -- this experiment
is NOT about training-set size, which RQ2 already covers; it's about how many synthetic
sequences you generate/release from that same fixed model).

For each replicate, draws one large synthetic pool (MAX_N) and computes the full
real(581) x synthetic(MAX_N) similarity matrix ONCE, then reads off DCR/exact-copy metrics
at each smaller release-size threshold N by slicing that matrix (nested by construction --
the size-1000 release is literally the first 1000 draws of the size-20000 release).

Metrics per N: mean DCR (synth->closest real... no: for each REAL individual, the distance to
its closest synthetic match, i.e. how "found" that person is by the release), the worst-case
tail (5th percentile of that distribution -- the most-exposed real individuals), and the
exact-copy count (cumulative, out of that many synthetic draws). Real-to-real leave-one-out
baseline is computed once (release-size independent) as the reference line.
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MATSUM_OUT", os.path.join(HERE, "data", "paris"))
import matsum_summarize as MS, matsum_synth as SY

M, K = 2, 5
MAX_N = 20000
NS = [200, 1000, 5000, 10000, 20000]
R = 5
REF_ALL = True   # use ALL 581 real individuals (not a REF_CAP subsample) -- tractable at this size

try:
    from scipy.stats import t as _t
    TCRIT = float(_t.ppf(0.975, df=R - 1))
except Exception:
    TCRIT = 2.776


def fast_sim_factory(symbols):
    S = len(symbols)
    Rm = np.zeros((S, S), dtype=bool)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            if b <= a:
                Rm[i, j] = True

    def sim(ida, idb):
        if len(ida) == 0 or len(idb) == 0:
            return 0.0
        M = Rm[np.ix_(ida, idb)]
        return (M.max(1).sum() + M.max(0).sum()) / (len(ida) + len(idb))
    return sim


print("[load] Paris-581")
gdf, areas = MS.load()
sem = MS.semantic_mapping(gdf, areas)
raw = SY.build_visits(sem, M)
seqs, support, _, _ = SY.anonymize_vocabulary(raw, K)
symbols = sorted({s for v in seqs.values() for s, _ in v}, key=lambda z: sorted(z))
sid = {s: i for i, s in enumerate(symbols)}
sim = fast_sim_factory(symbols)
real_ids = [np.array([sid[s] for s, _ in v]) for v in seqs.values()]
real_tuples = [tuple(v.tolist()) for v in real_ids]
real_tuple_set = set(real_tuples)
n_real = len(real_ids)
print(f"  {n_real} real individuals, {len(symbols)} symbols")

# real-to-real leave-one-out baseline (release-size independent)
t0 = time.time()
base_dcr = np.array([1 - max(sim(real_ids[i], real_ids[j]) for j in range(n_real) if j != i)
                     for i in range(n_real)])
print(f"  real-to-real baseline: mean={base_dcr.mean():.3f}  p5={np.percentile(base_dcr,5):.3f}"
      f"  ({time.time()-t0:.1f}s)")

# fit the generator ONCE (fixed model, matches the paper's m=2,k=5 operating point)
SY.rng = np.random.default_rng(42)
model = SY.SemanticMarkov(alpha=0.1).fit(seqs)

rows = []
for r in range(R):
    t0 = time.time()
    SY.rng = np.random.default_rng(1000 + r)
    synth = [SY.seq_symbols(model.sample(1000)) for _ in range(MAX_N)]
    synth_ids = [np.array([sid[s] for s in v]) for v in synth]
    print(f"  [rep {r+1}/{R}] generated {MAX_N} synthetic sequences ({time.time()-t0:.1f}s), "
          f"computing {n_real}x{MAX_N} similarity matrix...", flush=True)

    t0 = time.time()
    sim_matrix = np.zeros((n_real, MAX_N), dtype=np.float32)
    for i in range(n_real):
        ri = real_ids[i]
        for j in range(MAX_N):
            sim_matrix[i, j] = sim(ri, synth_ids[j])
        if (i + 1) % 100 == 0:
            print(f"    row {i+1}/{n_real}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"  [rep {r+1}/{R}] similarity matrix done ({time.time()-t0:.0f}s)", flush=True)

    for n in NS:
        max_sim_n = sim_matrix[:, :n].max(axis=1)
        dcr_n = 1 - max_sim_n
        exact_n = sum(1 for j in range(n) if tuple(synth_ids[j].tolist()) in real_tuple_set)
        rows.append(dict(rep=r, n=n, dcr_mean=float(dcr_n.mean()),
                          dcr_p5=float(np.percentile(dcr_n, 5)),
                          dcr_min=float(dcr_n.min()),
                          exact_copies=exact_n, exact_rate=exact_n / n))
    print(f"  replicate {r+1}/{R} done", flush=True)

import pandas as pd
raw_df = pd.DataFrame(rows)
OUT_DIR = os.environ["MATSUM_OUT"]; FIG_DIR = os.environ.get("MATSUM_FIG", OUT_DIR)
raw_df.to_csv(os.path.join(OUT_DIR, "paris_release_size_raw.csv"), index=False)

metrics = ["dcr_mean", "dcr_p5", "dcr_min", "exact_copies", "exact_rate"]
agg = []
for n in NS:
    sub = raw_df[raw_df.n == n]; row = {"n": n}
    for m in metrics:
        a = sub[m].to_numpy(float)
        row[m] = round(float(a.mean()), 4)
        row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
    agg.append(row)
adf = pd.DataFrame(agg)
adf.to_csv(os.path.join(OUT_DIR, "paris_release_size_summary.csv"), index=False)

pd.set_option("display.width", 220)
print(f"\n=== Release size vs. disclosure risk, Paris-581, fixed model, {R} replicates ===")
print(f"real-to-real baseline: DCR mean={base_dcr.mean():.4f}  p5={np.percentile(base_dcr,5):.4f}")
print(adf[["n", "dcr_mean", "dcr_mean_ci", "dcr_p5", "dcr_p5_ci", "exact_copies", "exact_rate"]].to_string(index=False))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
x = adf.n.to_numpy()
ax[0].errorbar(x, adf.dcr_mean, yerr=adf.dcr_mean_ci, fmt="o-", color="#2c7fb8", capsize=3, lw=2, label="mean DCR")
ax[0].errorbar(x, adf.dcr_p5, yerr=adf.dcr_p5_ci, fmt="s-", color="#d95f0e", capsize=3, lw=2, label="worst-case (p5) DCR")
ax[0].axhline(base_dcr.mean(), color="#555", ls="--", lw=1.5, label="real-to-real baseline (mean)")
ax[0].axhline(np.percentile(base_dcr, 5), color="#999", ls=":", lw=1.5, label="real-to-real baseline (p5)")
ax[0].set_title("Disclosure risk vs. release size"); ax[0].set_ylabel("DCR (up is safer)"); ax[0].legend(fontsize=8)
ax[1].errorbar(x, adf.exact_copies, yerr=adf.exact_copies_ci, fmt="D-", color="#6a53a6", capsize=3, lw=2)
ax[1].set_title("Cumulative exact copies"); ax[1].set_ylabel("exact-copy count (out of N released)")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("release size  N"); a.set_xticks(x); a.set_xticklabels(x, rotation=45)
fig.suptitle(f"Release size vs. disclosure risk -- Paris-581, fixed model, {R} replicates", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(FIG_DIR, "paris_release_size.png")
fig.savefig(p, dpi=130); plt.close(fig)
print("\nfigure:", p)
