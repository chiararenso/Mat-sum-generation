"""
RQ3 Pareto analysis: which generators are non-dominated on the fidelity / disclosure metrics?

Input: val_rq3_order2_summary.csv (means and 95% CIs over R=20 paired replicates for order-1 Markov,
order-2 Markov (light / heavy back-off) and LSTM at every n; built by val_rq3_order2.py from the same
paired samples as val_rq3.py). Per n (same data regime), a generator is Pareto-optimal if no other
generator is at least as good on every objective and strictly better on one.

Objectives: semantic bigram TV (min), semantic trigram TV (min), near-copy rate (min), dDCR (max).
We report (i) the 2-D fronts  [bigram TV, dDCR] and [trigram TV, dDCR], and (ii) the 4-D front.
Fronts use replicate means; the 95% CIs are drawn so that near-ties are visible rather than hidden.
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MATSUM_OUT", os.path.join(HERE, "data", "paris"))
FIG = os.environ.get("MATSUM_FIG", os.path.join(HERE, "val_paris", "figures"))
LABEL = {"Markov": "Markov-1", "Markov2": "Markov-2 (light)", "Markov2_heavy": "Markov-2 (heavy)", "LSTM": "LSTM"}
COL = {"Markov": "#2c7fb8", "Markov2": "#1b9e77", "Markov2_heavy": "#7fc97f", "LSTM": "#d95f0e"}
# (column, sign): sign=+1 means "minimise", -1 means "maximise" (we minimise sign*value)
OBJ4 = [("sem_genTV", 1), ("sem_triTV", 1), ("near_pct", 1), ("dDCR", -1)]


def pareto_mask(P):
    """P: (m, k) array to be MINIMISED. True for non-dominated rows."""
    m = len(P); keep = np.ones(m, bool)
    for i in range(m):
        for j in range(m):
            if i != j and np.all(P[j] <= P[i]) and np.any(P[j] < P[i]):
                keep[i] = False; break
    return keep


def main():
    df = pd.read_csv(os.path.join(OUT, "val_rq3_order2_summary.csv"))
    NS = sorted(df.n.unique()); models = ["Markov", "Markov2", "Markov2_heavy", "LSTM"]
    rows = []
    for n in NS:
        d = df[df.n == n].set_index("model").loc[models]
        fronts = {
            "bi_dDCR": pareto_mask(np.c_[d.sem_genTV.to_numpy(), -d.dDCR.to_numpy()]),
            "tri_dDCR": pareto_mask(np.c_[d.sem_triTV.to_numpy(), -d.dDCR.to_numpy()]),
            "all4": pareto_mask(np.c_[d.sem_genTV.to_numpy(), d.sem_triTV.to_numpy(),
                                       d.near_pct.to_numpy(), -d.dDCR.to_numpy()]),
        }
        for i, m in enumerate(models):
            rows.append({"n": n, "model": m, **{f"pareto_{k}": bool(v[i]) for k, v in fronts.items()},
                         **{c: d.loc[m, c] for c in ["sem_genTV", "sem_triTV", "near_pct", "dDCR"]}})
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "val_rq3_pareto.csv"), index=False)
    print("Pareto-optimal generators per n (replicate means):")
    for n in NS:
        r = res[res.n == n]
        print(f"  n={n:3d}  [bigram TV, dDCR]: {', '.join(LABEL[m] for m in r[r.pareto_bi_dDCR].model)}"
              f" | [trigram TV, dDCR]: {', '.join(LABEL[m] for m in r[r.pareto_tri_dDCR].model)}"
              f" | all 4: {', '.join(LABEL[m] for m in r[r.pareto_all4].model)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show_n = [20, 100, 581]
    fig, axes = plt.subplots(2, len(show_n), figsize=(13.5, 8.2), sharey="col")
    for c, n in enumerate(show_n):
        d = df[df.n == n].set_index("model").loc[models]
        for r_, (tcol, title) in enumerate([("sem_genTV", "bigram"), ("sem_triTV", "trigram")]):
            ax = axes[r_, c]
            P = np.c_[d[tcol].to_numpy(), -d.dDCR.to_numpy()]
            keep = pareto_mask(P)
            for i, m in enumerate(models):
                ax.errorbar(d.loc[m, tcol], d.loc[m, "dDCR"], xerr=d.loc[m, tcol + "_ci"], yerr=d.loc[m, "dDCR_ci"],
                            fmt="o", ms=11 if keep[i] else 8, color=COL[m], capsize=3, lw=1.4,
                            markeredgecolor="black" if keep[i] else COL[m], markeredgewidth=2 if keep[i] else 0,
                            label=LABEL[m] if (c == 0 and r_ == 0) else None, zorder=3)
            fp = P[keep]; order = np.argsort(fp[:, 0])
            ax.plot(fp[order, 0], -fp[order, 1], "--", color="#555", lw=1.2, zorder=1)
            ax.axhline(0, color="#999", lw=0.8, ls=":")
            ax.set_xlabel(f"semantic {title} TV  (↓ better)")
            if c == 0:
                ax.set_ylabel("ΔDCR  (↑ safer)")
            ax.set_title(f"n = {n}" if r_ == 0 else "", fontsize=12)
    axes[0, 0].legend(fontsize=9, loc="lower left")
    fig.suptitle("RQ3 · Pareto view of fidelity vs disclosure (Paris-OSM, 20 paired replicates, mean ± 95% CI)\n"
                 "black outline + dashed line: Pareto-optimal generators on that pair of metrics", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIG, exist_ok=True); p = os.path.join(FIG, "val_rq3_pareto.png")
    fig.savefig(p, dpi=130); plt.close(fig); print("figure:", p)


if __name__ == "__main__":
    main()
