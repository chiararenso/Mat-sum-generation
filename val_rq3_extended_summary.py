"""
Summaries of the extended RQ3 runs written by val_rq3_extended.py:
  tag 'ngrams'      : markov1, markov2l, markov2h, lstm_h64 -> TV for n-gram orders 1..5 (+ real-vs-real floor)
  tags h128/h256/l2 : LSTM capacity scale (hidden 128, hidden 256, 2 layers) -> TV orders 1..5, near-copy, dDCR, exact copies
Outputs (data/paris): val_rq3_ext_ngrams_summary.csv, val_rq3_ext_ngrams_paired.csv,
                      val_rq3_ext_scale_summary.csv, val_rq3_ext_scale_paired.csv
The original LSTM (hidden 64, 1 layer) row of the scale comparison is taken from the regenerated 'ngrams' run for the
n-gram TVs and from results/paris/val_rq3_raw.csv for near-copy / dDCR / exact copies (same samples and seeds).
"""
import os, numpy as np, pandas as pd
from scipy.stats import t as _t, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MATSUM_OUT", os.path.join(HERE, "data", "paris"))
NS = [20, 50, 100, 300, 581]
LAB = {"markov1": "Markov-1", "markov2l": "Markov-2 light", "markov2h": "Markov-2 heavy", "lstm_h64": "LSTM h64 (original)",
       "lstm_h128": "LSTM h128", "lstm_h256": "LSTM h256", "lstm_l2": "LSTM 2 layers (h64)", "real-vs-real": "real-vs-real"}


def mci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return float(a.mean()), float(_t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))


def paired(d, a, b, cols):
    rows = []
    for n in NS:
        x = d[(d.n == n) & (d.model == a)].sort_values("rep"); y = d[(d.n == n) & (d.model == b)].sort_values("rep")
        if len(x) != len(y) or len(x) == 0:
            continue
        for c in cols:
            diff = x[c].to_numpy(float) - y[c].to_numpy(float)
            rows.append({"n": n, "model": a, "vs": b, "metric": c, "d_mean": float(diff.mean()),
                         "a_better_of_reps": int((diff < 0).sum()) if not c.startswith("dDCR") else int((diff > 0).sum()),
                         "reps": len(diff), "wilcoxon_p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0})
    return rows


def main():
    ng = pd.read_csv(os.path.join(OUT, "val_rq3_ext_ngrams_raw.csv"))
    tv = [f"TV{k}" for k in range(1, 6)]
    summ = []
    for (n, m), g in ng.groupby(["n", "model"]):
        row = {"n": n, "model": m}
        for c in tv:
            row[c], row[c + "_ci"] = mci(g[c])
        summ.append(row)
    pd.DataFrame(summ).sort_values(["n", "model"]).to_csv(os.path.join(OUT, "val_rq3_ext_ngrams_summary.csv"), index=False)
    pr = paired(ng, "markov2l", "lstm_h64", tv[1:]) + paired(ng, "markov2h", "lstm_h64", tv[1:]) + paired(ng, "markov2l", "markov1", tv[1:])
    pd.DataFrame(pr).to_csv(os.path.join(OUT, "val_rq3_ext_ngrams_paired.csv"), index=False)
    print("saved n-gram summary + paired")

    import glob
    parts = sorted(sum([glob.glob(os.path.join(OUT, f"val_rq3_ext_{t}*_raw.csv")) for t in ("h128", "h256", "l2")], []))
    if not parts:
        return
    sc = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    sc = sc[sc.model != "real-vs-real"].drop_duplicates(["rep", "n", "model"], keep="first")
    old = pd.read_csv(os.path.join(HERE, "results", "paris", "val_rq3_raw.csv"))
    old = old[old.model == "LSTM"].rename(columns={"near_pct": "near_pct", "dDCR": "dDCR"})
    base = ng[ng.model == "lstm_h64"].merge(old[["rep", "n", "near_pct", "dDCR", "exact_copies"]], on=["rep", "n"], how="left")
    o2 = pd.read_csv(os.path.join(OUT, "val_rq3_order2_raw.csv")).rename(columns={"model": "m_old"})
    o2l = ng[ng.model == "markov2l"].merge(o2[o2.m_old == "Markov2"][["rep", "n", "near_pct", "dDCR", "exact_copies"]], on=["rep", "n"], how="left")
    allm = pd.concat([sc, base, o2l], ignore_index=True)
    cols = tv + ["near_pct", "dDCR", "exact_copies"]
    out = []
    for (n, m), g in allm.groupby(["n", "model"]):
        row = {"n": n, "model": m, "reps": g.rep.nunique()}
        for c in cols:
            row[c], row[c + "_ci"] = mci(g[c])
        out.append(row)
    pd.DataFrame(out).sort_values(["n", "model"]).to_csv(os.path.join(OUT, "val_rq3_ext_scale_summary.csv"), index=False)
    pr = []
    for lstm in ["lstm_h128", "lstm_h256", "lstm_l2"]:
        if lstm in set(allm.model):
            pr += paired(allm, lstm, "lstm_h64", ["TV2", "TV3", "TV4", "TV5", "near_pct", "dDCR"])
            pr += paired(allm, "markov2l", lstm, ["TV2", "TV3", "TV4", "TV5", "near_pct", "dDCR"])
    pd.DataFrame(pr).to_csv(os.path.join(OUT, "val_rq3_ext_scale_paired.csv"), index=False)
    print("saved capacity-scale summary + paired (replicates per config):", allm.groupby("model").rep.nunique().to_dict())


if __name__ == "__main__":
    main()


def figure():
    """Capacity scale: LSTM hidden 64 / 128 / 256 (and 2 layers) vs the order-2 Markov (light) reference, for n = 20, 100, 581."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = pd.read_csv(os.path.join(OUT, "val_rq3_ext_scale_summary.csv"))
    fig_dir = os.environ.get("MATSUM_FIG", os.path.join(HERE, "val_paris", "figures")); os.makedirs(fig_dir, exist_ok=True)
    cfg = [("lstm_h64", "h64"), ("lstm_h128", "h128"), ("lstm_h256", "h256"), ("lstm_l2", "2 layers\n(h64)")]
    panels = [("TV3", "(A) trigram TV (↓)"), ("TV5", "(B) 5-gram TV (↓)"), ("near_pct", "(C) near-copy % (↓)"), ("dDCR", "(D) ΔDCR (↑)")]
    col = {20: "#d95f0e", 100: "#2c7fb8", 581: "#1b9e77"}
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.3)); x = np.arange(len(cfg))
    for a, (c, title) in zip(ax, panels):
        for n in (20, 100, 581):
            t = s[s.n == n].set_index("model")
            a.errorbar(x, [t.loc[m, c] for m, _ in cfg], yerr=[t.loc[m, c + "_ci"] for m, _ in cfg], fmt="o-", color=col[n], capsize=3, lw=1.8,
                       label=f"LSTM, n={n}")
            a.axhline(t.loc["markov2l", c], color=col[n], ls="--", lw=1.2, alpha=0.8)
        a.set_xticks(x); a.set_xticklabels([l for _, l in cfg]); a.set_title(title, fontsize=11)
        if c == "near_pct":
            a.set_yscale("symlog", linthresh=1)
        if c == "dDCR":
            a.axhline(0, color="#888", lw=0.8, ls=":")
    ax[0].legend(fontsize=8)
    fig.suptitle("RQ3 · LSTM capacity scale (Paris-OSM, 20 paired replicates, mean ± 95% CI). Dashed lines: order-2 Markov (light back-off), same colour = same n", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); p = os.path.join(fig_dir, "val_rq3_capacity_scale.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("figure:", p)


if __name__ == "__main__":
    figure()
