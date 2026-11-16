"""
Redraws the matched-budget figure (grid / cluster / semantic states, same order-1 Markov) from the saved results
val_abstraction_budget.csv written by val_abstraction_budget.py. No experiment is re-run.
Labels: the semantic representation is shown as "Semantic-Markov".
Env: MATSUM_OUT (folder with the csv, default data/paris), MATSUM_FIG (output folder, default figures).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MATSUM_OUT", os.path.join(HERE, "data", "paris"))
FIG = os.environ.get("MATSUM_FIG", os.path.join(HERE, "figures"))
R = 20  # paired samples behind the confidence intervals

LABEL = {"Grid-Markov": "Grid-Markov", "Cluster-Markov": "Cluster-Markov", "MAT-Sum-Markov": "Semantic-Markov",
         "Top1-label-Markov": "Top-1-label-Markov"}
STYLE = {"Grid-Markov": ("#8aa0b3", "o"), "Cluster-Markov": ("#2c7fb8", "s"), "MAT-Sum-Markov": ("#d95f0e", "D")}
TOP1_COLOR = "#6a3d9a"
PANELS = [("genTV2", "(A) Fidelity, bigram (label space)\nTop-1 is native to this metric (see text)",
           "semantic generation TV, bigram (lower is better)"),
          ("genTV3", "(B) Fidelity, trigram (label space)\nsame caveat for Top-1",
           "semantic generation TV, trigram (lower is better)"),
          ("copies", "(C) Memorization\nfine resolution at about 0 copies; Top-1 is coarse",
           "exact copies of real records (out of 200)")]


def main():
    df = pd.read_csv(os.path.join(OUT, "val_abstraction_budget.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (metric, title, ylab) in zip(axes, PANELS):
        for m, (col, mk) in STYLE.items():
            d = df[(df.method == m) & (df.metric == metric)].sort_values("states")
            ax.errorbar(d["states"], d["mean"], yerr=d["ci95"], marker=mk, ls="-", color=col, capsize=3, lw=2, ms=6,
                        label=LABEL[m])
        t = df[(df.method == "Top1-label-Markov") & (df.metric == metric)]
        ax.errorbar(t["states"], t["mean"], yerr=t["ci95"], marker="*", ls="none", color=TOP1_COLOR, capsize=3, ms=16,
                    label=LABEL["Top1-label-Markov"] + " (trivial)")
        ax.set_xscale("log")
        ax.set_xlabel("number of states (matched budget, log)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Advantage at matched budget · Paris-581 · same order-1 Markov · %d paired samples (95%% CI)" % R,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "fig_abstraction_budget_semantic.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("figure:", p)


if __name__ == "__main__":
    main()
