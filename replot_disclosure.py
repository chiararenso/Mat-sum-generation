"""Re-plot disclosure tail analysis from the saved raw npz (max-MUITAS arrays), at an explicit tau grid
that includes 0.95. Usage: python replot_disclosure.py <dataset> <npz> <ns-csv> <fig_dir>."""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

name, npz, ns_s, figdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
NS = [int(x) for x in ns_s.split(",")]
TAUS = np.array([0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.0])
d = np.load(npz)

rows = []
for n in NS:
    syn = d[f"synth_{n}"]; rel = d[f"real_{n}"]; dcr = 1 - syn
    row = {"n": n, "DCR_mean": float(dcr.mean()), "DCR_p05": float(np.percentile(dcr, 5)),
           "DCR_min": float(dcr.min()), "pct_DCR_lt0.10": float(100 * (dcr < 0.10).mean()),
           "pct_DCR_lt0.05": float(100 * (dcr < 0.05).mean())}
    for t in TAUS:
        row[f"synth@{t}"] = float(100 * (syn >= t).mean())
        row[f"real@{t}"] = float(100 * (rel >= t).mean())
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv(os.path.join(figdir, f"{name}_disclosure.csv"), index=False)
pd.set_option("display.width", 220)
print(f"\n=== {name} DCR tail ===")
print(df[["n", "DCR_mean", "DCR_p05", "DCR_min", "pct_DCR_lt0.10", "pct_DCR_lt0.05"]].to_string(index=False))
print(f"\n=== {name} near-copy(tau) synth vs real baseline (%) ===")
show = ["n"] + [f"synth@{t}" for t in (0.9, 0.95, 1.0)] + [f"real@{t}" for t in (0.9, 0.95, 1.0)]
print(df[show].to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(NS)))
for c, n in zip(cmap, NS):
    s = df[df.n == n]
    ax[0].plot(TAUS, [s[f"synth@{t}"].iloc[0] for t in TAUS], "-o", color=c, lw=2, ms=4, label=f"synth n={n}")
    ax[0].plot(TAUS, [s[f"real@{t}"].iloc[0] for t in TAUS], "--", color=c, lw=1.2, alpha=.7)
ax[0].axvline(0.95, color="#bbb", ls=":", lw=1)
ax[0].set_xlabel("near-copy threshold τ (MUITAS)"); ax[0].set_ylabel("% records ≥ τ (↓ safer)")
ax[0].set_title("(A) Near-copy rate vs threshold\nsolid = synth→train · dashed = real→real baseline")
ax[0].legend(fontsize=8); ax[0].set_yscale("symlog", linthresh=0.5)
ax[1].plot(df.n, df.DCR_mean, "o-", color="#d95f0e", lw=2, label="mean DCR")
ax[1].plot(df.n, df.DCR_p05, "s--", color="#2c7fb8", lw=2, label="5th-percentile DCR (tail)")
ax[1].plot(df.n, df.DCR_min, "^:", color="#555", lw=1.5, label="min DCR (worst case)")
ax[1].set_xscale("log"); ax[1].set_xticks(NS); ax[1].set_xticklabels(NS); ax[1].minorticks_off()
ax[1].set_xlabel("number of individuals  n"); ax[1].set_ylabel("DCR = 1 − max MUITAS (↑ safer)")
ax[1].set_title("(B) DCR distribution: mean vs tail"); ax[1].legend(fontsize=9)
fig.suptitle(f"{name} · disclosure tail analysis · MAT-Sum+Markov · 20 replicates", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(figdir, f"{name}_disclosure.png"); fig.savefig(p, dpi=130); plt.close(fig)
print("\nfigure:", p)
