"""
LaTeX tables for the extended RQ3 analysis (n-gram orders up to 5, LSTM capacity scale), generated from
data/paris/val_rq3_ext_{ngrams,scale}_{summary,paired}.csv  ->  paper_tables/rq3_extended.tex
"""
import os, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "paris")
OUT = os.path.join(HERE, "paper_tables"); os.makedirs(OUT, exist_ok=True)
ng = pd.read_csv(os.path.join(D, "val_rq3_ext_ngrams_summary.csv")); ngp = pd.read_csv(os.path.join(D, "val_rq3_ext_ngrams_paired.csv"))
sc = pd.read_csv(os.path.join(D, "val_rq3_ext_scale_summary.csv"))
neg = lambda s: s.replace("-", "$-$") if s.startswith("-") else s
f3 = lambda m, c: neg(f"{m:.3f}") + "$\\pm$" + f"{c:.3f}"
f2 = lambda m, c: neg(f"{m:.2f}") + "$\\pm$" + f"{c:.2f}"


def stars(p):
    return "$^{***}$" if p < 1e-3 else ("$^{**}$" if p < 1e-2 else ("$^{*}$" if p < 5e-2 else ""))


CFG = [("lstm_h64", "LSTM, hidden 64 (original)"), ("lstm_h128", "LSTM, hidden 128"), ("lstm_h256", "LSTM, hidden 256"),
       ("lstm_l2", "LSTM, 2 layers (hidden 64)"), ("markov2l", "Markov-2 (light)")]
o = []; w = o.append

# ---- main table: capacity scale at n = 20 and n = 581
w(r"""\begin{table}[!t]
\caption{RQ3 on Paris-OSM: LSTM capacity scale against the light order-2 Markov (mean $\pm$ 95\% CI over 20 paired replicates; all other settings as in Table~\ref{tab:order2}). Near-copy: \% of 200 synthetic sequences with MUITAS $\ge 0.95$. Lower TV and near-copy, higher $\Delta$DCR, are better.}
\label{tab:capacity}
\centering\scriptsize
\begin{tabular}{l cccc cccc}
\toprule
 & \multicolumn{4}{c}{$n{=}20$} & \multicolumn{4}{c}{$n{=}581$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
generator & TV$_3$ & TV$_5$ & near-copy & $\Delta$DCR & TV$_3$ & TV$_5$ & near-copy & $\Delta$DCR \\
\midrule""")
for m, lab in CFG:
    cells = []
    for n in (20, 581):
        t = sc[(sc.n == n) & (sc.model == m)].iloc[0]
        cells += [f3(t.TV3, t.TV3_ci), f3(t.TV5, t.TV5_ci), f2(t.near_pct, t.near_pct_ci), f3(t.dDCR, t.dDCR_ci)]
    w(f"{lab} & " + " & ".join(cells) + r" \\")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- supplementary: n-gram TV of the RQ3 generators
w(r"""\section{RQ3: $n$-gram fidelity up to order 5}
\label{supp:rq3-ngrams}
Table~\ref{tab:supp-rq3-ngrams} reports the TV between the real and the synthetic $n$-gram distributions in the common OSM-label space for orders 1--5, for the order-1 Markov, the order-2 Markov (light and heavy back-off), the original LSTM, and the real-vs-real floor (two disjoint halves of the real sample). The synthetic sequences are regenerated with the seeds of the original RQ3 runs; the LSTM and order-2 rows reproduce the values of Tables~\ref{tab:supp-order2-full} exactly. Table~\ref{tab:supp-rq3-ngrams-paired} gives the paired differences between each order-2 variant and the LSTM (negative: the order-2 model is better).

\begin{table}[!t]
\caption{$n$-gram TV in the common OSM-label space on Paris-OSM (mean $\pm$ 95\% CI over 20 paired replicates; lower is better). Floor: two disjoint halves of the real sample.}
\label{tab:supp-rq3-ngrams}
\centering\small
\begin{tabular}{llccccc}
\toprule
$n$ & generator & order 1 & order 2 & order 3 & order 4 & order 5 \\
\midrule""")
LAB = {"markov1": "Markov-1", "markov2l": "Markov-2 (light)", "markov2h": "Markov-2 (heavy)", "lstm_h64": "LSTM", "real-vs-real": "real-vs-real floor"}
for n in (20, 100, 581):
    for i, m in enumerate(["markov1", "markov2l", "markov2h", "lstm_h64", "real-vs-real"]):
        t = ng[(ng.n == n) & (ng.model == m)].iloc[0]
        w(f"{n if i == 0 else ''} & {LAB[m]} & " + " & ".join(f3(t[f"TV{k}"], t[f"TV{k}_ci"]) for k in range(1, 6)) + r" \\")
    if n != 581:
        w(r"\midrule")
w(r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Paired differences in $n$-gram TV, order-2 variant minus LSTM (mean over 20 paired replicates; negative favours the order-2 model). Stars: paired Wilcoxon $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}
\label{tab:supp-rq3-ngrams-paired}
\centering\small
\begin{tabular}{llcccc}
\toprule
variant & $n$ & order 2 & order 3 & order 4 & order 5 \\
\midrule""")
for v, lab in [("markov2l", "Markov-2 (light)"), ("markov2h", "Markov-2 (heavy)")]:
    for n in (20, 50, 100, 300, 581):
        cells = []
        for k in (2, 3, 4, 5):
            r = ngp[(ngp.model == v) & (ngp.vs == "lstm_h64") & (ngp.n == n) & (ngp.metric == f"TV{k}")].iloc[0]
            cells.append(neg(f"{r.d_mean:.3f}") + stars(r.wilcoxon_p))
        w(f"{lab if n == 20 else ''} & {n} & " + " & ".join(cells) + r" \\")
    if v == "markov2l":
        w(r"\midrule")
w(r"""\bottomrule
\end{tabular}
\end{table}

\section{RQ3: LSTM capacity scale}
\label{supp:rq3-capacity}
To check that the LSTM comparison does not depend on a single small network, we widen the LSTM (hidden size 128 and 256, one layer) and deepen it (two layers of size 64), keeping embedding size 32, the optimiser (Adam, learning rate $5\times10^{-3}$), the 120 full-batch epochs, the vocabulary, the nested samples and the seeds of RQ3 fixed. Table~\ref{tab:supp-capacity} reports all five sample sizes. Larger networks lower the 4- and 5-gram TV but raise the bigram TV and, above all, the near-copy rate and the proximity to the training records when data are scarce; two layers bring no gain. No configuration matches the light order-2 Markov on fidelity and disclosure together. These results hold for the LSTM family and the fixed training protocol used here; no hyperparameter search, dropout or early stopping was tried.

\begin{table}[!t]
\caption{RQ3 on Paris-OSM: LSTM capacity scale against the light order-2 Markov (mean $\pm$ 95\% CI over 20 paired replicates). Near-copy: \% of 200 synthetic sequences with MUITAS $\ge 0.95$.}
\label{tab:supp-capacity}
\centering\scriptsize
\begin{tabular}{llccccc}
\toprule
$n$ & generator & TV$_2$ & TV$_3$ & TV$_5$ & near-copy \% & $\Delta$DCR \\
\midrule""")
for n in (20, 50, 100, 300, 581):
    for i, (m, lab) in enumerate(CFG):
        t = sc[(sc.n == n) & (sc.model == m)].iloc[0]
        w(f"{n if i == 0 else ''} & {lab} & {f3(t.TV2, t.TV2_ci)} & {f3(t.TV3, t.TV3_ci)} & {f3(t.TV5, t.TV5_ci)} & {f2(t.near_pct, t.near_pct_ci)} & {f3(t.dDCR, t.dDCR_ci)} \\\\")
    if n != 581:
        w(r"\midrule")
w(r"""\bottomrule
\end{tabular}
\end{table}""")
open(os.path.join(OUT, "rq3_extended.tex"), "w").write("\n".join(o))
print(len(o), "lines ->", os.path.join(OUT, "rq3_extended.tex"))
