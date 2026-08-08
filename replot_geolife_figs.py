"""Regenerate the GeoLife E1/E2/E3 figures from the committed summary CSVs, with clean titles
(E1/E2/E3 -> GeoLife (RQ1/RQ2/RQ3)). No experiment re-run: reads results/geolife/*_summary.csv.
Layouts/colors reproduce the original scripts exactly; only the suptitle text changes."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES = "results/geolife"; FIG = "paper/figures"; R = 20
os.makedirs(FIG, exist_ok=True)

# ---------- E1 -> RQ1 ----------
adf = pd.read_csv(f"{RES}/geolife_e1_summary.csv")
x = np.arange(len(adf)); fig, (aA, aB) = plt.subplots(1, 2, figsize=(11, 4.7))
aA.bar(x, adf.native_biTV, 0.55, yerr=adf.native_biTV_ci, capsize=4, color="#2c7fb8")
aA.set_xticks(x); aA.set_xticklabels(adf.method); aA.set_ylabel("TV (↓ better)")
aA.set_title("(A) native-state bigram TV")
w = 0.38
aB.bar(x - w/2, adf.distortion, w, yerr=adf.distortion_ci, capsize=4, color="#8aa0b3", label="abstraction distortion")
aB.bar(x + w/2, adf.sem_genTV, w, yerr=adf.sem_genTV_ci, capsize=4, color="#d95f0e", label="generation (synth→labels)")
aB.set_xticks(x); aB.set_xticklabels(adf.method); aB.set_title("(B) common-label-space TV"); aB.legend(fontsize=9)
fig.suptitle(f"GeoLife (RQ1) · role of the abstraction · ~{int(adf.n_states.mean())} states · n=100 · {R} paired samples (95% CI)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{FIG}/fig_geolife_e1.png", dpi=130); plt.close(fig)
print("wrote fig_geolife_e1.png")

# ---------- E2 -> RQ2 ----------
adf = pd.read_csv(f"{RES}/geolife_e2_summary.csv").sort_values("n"); ns = adf.n.to_numpy(); sat = 75
x = ns; fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
ax[0].errorbar(x, adf.sem_genTV, yerr=adf.sem_genTV_ci, fmt="o-", color="#2c7fb8", capsize=3, lw=2)
ax[0].axvline(sat, color="#888", ls=":", label=f"n*≈{sat}"); ax[0].legend(fontsize=9)
ax[0].set_title("(A) Fidelity"); ax[0].set_ylabel("semantic generation TV (↓)")
ax[1].errorbar(x, adf.DCR_synth, yerr=adf.DCR_synth_ci, fmt="s-", color="#d95f0e", capsize=3, lw=2, label="synth→training")
ax[1].errorbar(x, adf.DCR_base, yerr=adf.DCR_base_ci, fmt="o--", color="#555", capsize=3, lw=2, label="real→real (user-aware)")
ax[1].set_title("(B) Distance to real records"); ax[1].set_ylabel("DCR (↑ safer)"); ax[1].legend(fontsize=9)
ax[2].errorbar(x, adf.exact_copies, yerr=adf.exact_copies_ci, fmt="D-", color="#6a53a6", capsize=3, lw=2)
ax[2].set_title("(C) Exact session copies"); ax[2].set_ylabel("exact copies (of 200)"); ax[2].axhline(0, color="#ccc", lw=.8)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off()
fig.suptitle(f"GeoLife (RQ2) · MAT-Sum+Markov scaling · {R} nested paired replicates · fixed abstraction parameters", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{FIG}/fig_geolife_e2.png", dpi=130); plt.close(fig)
print("wrote fig_geolife_e2.png")

# ---------- E3 -> RQ3 ----------
adf = pd.read_csv(f"{RES}/geolife_e3_summary.csv"); ns = sorted(adf.n.unique())
x = np.array(ns); fig, ax = plt.subplots(1, 3, figsize=(14, 4.4)); C = {"Markov": "#2c7fb8", "LSTM": "#d95f0e"}
for g in ("Markov", "LSTM"):
    s = adf[adf.gen == g].sort_values("n")
    ax[0].errorbar(x, s.sem_biTV, yerr=s.sem_biTV_ci, fmt="o-", color=C[g], capsize=3, lw=2, label=g)
    ax[1].errorbar(x, s.near_pct, yerr=s.near_pct_ci, fmt="s-", color=C[g], capsize=3, lw=2, label=g)
    ax[2].errorbar(x, s.dDCR, yerr=s.dDCR_ci, fmt="D-", color=C[g], capsize=3, lw=2, label=g)
ax[0].set_title("(A) Fidelity"); ax[0].set_ylabel("semantic generation TV (↓)")
ax[1].set_title("(B) Near-copy disclosure"); ax[1].set_ylabel("% synth with max MUITAS ≥ 0.95 (↓)")
ax[2].set_title("(C) ΔDCR (↑ safer)"); ax[2].set_ylabel("DCR synth − real baseline"); ax[2].axhline(0, color="#ccc", lw=.8)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off(); a.legend(fontsize=9)
fig.suptitle(f"GeoLife (RQ3) · generator capacity (Markov vs LSTM) · same MAT-Sum representation · {R} nested paired replicates", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{FIG}/geolife_e3.png", dpi=130); plt.close(fig)
print("wrote geolife_e3.png")
