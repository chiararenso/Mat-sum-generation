"""
Generates the LaTeX tables of the RQ1 n-gram / RQ3 order-2 / Pareto / hypervolume additions from the
result CSVs in data/paris/ (val_abstraction_ngrams*.csv, val_rq3_order2_*.csv, val_rq3_pareto.csv,
val_rq3_hypervolume_summary.csv). Output (git-tracked, regenerable): paper_tables/supp_additions.tex
(four supplementary subsections) and paper_tables/main_order2_table.tex (Section 4.5 table).
Run the experiment scripts first:  val_abstraction_ngrams.py, val_rq3_order2.py, val_rq3_pareto.py,
val_rq3_hypervolume.py   then   python3 make_supp_tex.py
"""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "paris") + os.sep
OUTDIR = os.path.join(HERE, "paper_tables")
os.makedirs(OUTDIR, exist_ok=True)
LAB = {"Markov": "Markov-1", "Markov2": "Markov-2 (light)", "Markov2_heavy": "Markov-2 (heavy)", "LSTM": "LSTM"}
MODELS = ["Markov", "Markov2", "Markov2_heavy", "LSTM"]
NS = [20, 50, 100, 300, 581]
neg = lambda s: s.replace("-", "$-$") if s.startswith("-") else s


from decimal import Decimal, ROUND_HALF_UP


def hu(x, d):
    return str(Decimal(str(round(float(x), 8))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pm(m, c, d=3):
    # TV / dDCR: plain 3-decimal formatting of the 4-decimal CSV values (reproduces Table 7 of the paper);
    # near-copy / exact copies (2 decimals): round half up, so 0.475 reads 0.48
    f = (lambda x: hu(x, d)) if d == 2 else (lambda x: f"{x:.{d}f}")
    return neg(f(m)) + "$\\pm$" + f(c)


def stars(p):
    return "$^{***}$" if p < 1e-3 else ("$^{**}$" if p < 1e-2 else ("$^{*}$" if p < 5e-2 else ""))


out = []
w = out.append

# ---------------- S-A: n-grams ----------------
ng = pd.read_csv(D + "val_abstraction_ngrams.csv")
pg = pd.read_csv(D + "val_abstraction_ngrams_paired.csv")
w(r"""\subsection{Robustness of RQ1 beyond bigrams}
\label{supp:ngrams}
Table~\ref{tab:supp-ngram-paired} reports the paired differences $d_i=\mathrm{TV}_{i,\text{baseline}}-\mathrm{TV}_{i,\text{MAT-Sum}}$ underlying the $n$-gram comparison of Section~4.3 (positive values favour MAT-Sum), computed over the same $R{=}20$ paired user samples as Tables~3--4. Table~\ref{tab:supp-ngram-native} gives the corresponding TV in each representation's native state space; the split-half floor is computed in the same state space. The real-vs-real floor is the TV between two disjoint random halves of the paired real sample, each of about the size of the synthetic release, and indicates the order beyond which $n$-gram estimates are dominated by sampling sparsity (distinct real label $n$-grams in a paired sample: 57, 735, 3{,}199, 6{,}307 and 8{,}421 for $n{=}1,\dots,5$).
""")
w(r"""\begin{table}[!t]
\caption{Paired differences in label-space $n$-gram TV (baseline $-$ MAT-Sum; mean over 20 paired samples), Cohen's $d_z$, and number of samples (out of 20) in which MAT-Sum is better. All Wilcoxon $p=1.9\times10^{-6}$ unless noted.}
\label{tab:supp-ngram-paired}
\centering\small
\begin{tabular}{l ccc ccc ccc}
\toprule
 & \multicolumn{3}{c}{vs.\ Grid} & \multicolumn{3}{c}{vs.\ Cluster} & \multicolumn{3}{c}{vs.\ Top-1 label} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
$n$ & $d$ & $d_z$ & wins & $d$ & $d_z$ & wins & $d$ & $d_z$ & wins \\
\midrule""")
for n in [1, 2, 3, 4, 5]:
    cells = []
    for b in ["Grid-Markov", "Cluster-Markov", "Top1-Markov"]:
        r = pg[(pg.n == n) & (pg.baseline == b)].iloc[0]
        extra = "" if r.wilcoxon_p < 1e-5 else f"$^\\dagger$"
        cells += [neg(f"{r.d_mean:+.3f}").replace("+", "") if r.d_mean > 0 else neg(f"{r.d_mean:.3f}"),
                  neg(f"{r.d_z:.1f}"), f"{int(r.matsum_wins_of_20)}/20{extra}"]
    w(f"{n} & " + " & ".join(cells) + r" \\")
w(r"""\bottomrule
\multicolumn{10}{l}{\footnotesize $^\dagger$ not significant ($p=0.62$ for $n{=}4$ vs.\ Top-1 label).}
\end{tabular}
\end{table}
""")
w(r"""\begin{table}[!t]
\caption{Native-state $n$-gram TV (each representation vs.\ its own real distribution; mean $\pm$ 95\% CI) with the real-vs-real split-half floor in the same state space (in brackets). Lower is better; values close to 1 indicate saturation.}
\label{tab:supp-ngram-native}
\centering\small
\begin{tabular}{lccc}
\toprule
 & $n{=}2$ & $n{=}3$ & $n{=}4$ \\
\midrule""")
for m, lab in [("Grid-Markov", "Grid"), ("Cluster-Markov", "Cluster"), ("MAT-Sum-Markov", "MAT-Sum"), ("Top1-Markov", "Top-1 label")]:
    cells = []
    for n in [2, 3, 4]:
        r = ng[(ng.space == "native") & (ng.n == n) & (ng.method == m)].iloc[0]
        cells.append(f"{r['mean']:.3f}$\\pm${r.ci95:.3f} [{r.floor_mean:.3f}]")
    w(f"{lab} & " + " & ".join(cells) + r" \\")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S-B: order-2 ----------------
sm = pd.read_csv(D + "val_rq3_order2_summary.csv")
pr = pd.read_csv(D + "val_rq3_order2_paired.csv")
w(r"""\subsection{RQ3 capacity ladder: order-2 Markov}
\label{supp:order2}
The order-2 generator is $P(j\mid a,b)=(C_{abj}+\lambda P_1(j\mid b))/(C_{ab}+\lambda)$, with $P_1$ the smoothed order-1 model ($\alpha{=}0.1$), unseen contexts falling back exactly to $P_1$; the light variant uses $\lambda{=}1$ and the heavy variant $\lambda{=}\alpha|\mathcal{V}_{m,k}|$. Order-1 Markov and LSTM rows are those of Table~7 and Figure~7 (same nested paired samples, same reference sets and matched real-to-real baseline); the order-2 rows reuse them, so all comparisons are paired by (replicate, $n$). Table~\ref{tab:supp-order2-full} reports all metrics at every $n$, and Table~\ref{tab:supp-order2-paired} the paired differences.
""")
w(r"""\begin{table}[!t]
\caption{Capacity ladder on Paris-OSM (MAT-Sum, $m{=}2$, $k{=}5$; 20 nested paired replicates; mean $\pm$ 95\% CI; $N{=}200$ synthetic sequences per run). Near-copy: \% with MUITAS $\ge 0.95$; exact copies out of 200.}
\label{tab:supp-order2-full}
\centering\small
\begin{tabular}{llccccc}
\toprule
$n$ & generator & bigram TV & trigram TV & near-copy \% & $\Delta$DCR & exact \\
\midrule""")
for n in NS:
    for i, m in enumerate(MODELS):
        r = sm[(sm.n == n) & (sm.model == m)].iloc[0]
        first = str(n) if i == 0 else ""
        w(f"{first} & {LAB[m]} & {pm(r.sem_genTV, r.sem_genTV_ci)} & {pm(r.sem_triTV, r.sem_triTV_ci)} & "
          f"{pm(r.near_pct, r.near_pct_ci, 2)} & {pm(r.dDCR, r.dDCR_ci)} & {pm(r.exact_copies, r.exact_copies_ci, 2)} \\\\")
    if n != NS[-1]:
        w(r"\midrule")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")
w(r"""\begin{table}[!t]
\caption{Paired differences, order-2 variant minus comparator (mean over 20 paired replicates). TV and near-copy: negative favours the order-2 model; $\Delta$DCR: positive favours it. Stars: paired Wilcoxon $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}
\label{tab:supp-order2-paired}
\centering\small
\begin{tabular}{llccccc}
\toprule
variant & vs. & $n$ & $\Delta$ bigram TV & $\Delta$ trigram TV & $\Delta$ near-copy (pp) & $\Delta\Delta$DCR \\
\midrule""")
blocks = [("Markov2", "Markov"), ("Markov2", "LSTM"), ("Markov2_heavy", "Markov"), ("Markov2_heavy", "LSTM")]
for bi, (mod, comp) in enumerate(blocks):
    for n in NS:
        cells = []
        for met in ["sem_genTV", "sem_triTV", "near_pct", "dDCR"]:
            r = pr[(pr.model == mod) & (pr.vs == comp) & (pr.n == n) & (pr.metric == met)].iloc[0]
            val = f"{r.d_mean:.3f}" if met != "near_pct" else f"{r.d_mean:.2f}"
            cells.append(neg(val) + stars(r.wilcoxon_p))
        lead = (LAB[mod] if n == NS[0] else "")
        lead2 = ("Markov-1" if comp == "Markov" else "LSTM") if n == NS[0] else ""
        w(f"{lead} & {lead2} & {n} & " + " & ".join(cells) + r" \\")
    if bi != len(blocks) - 1:
        w(r"\midrule")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S-C: Pareto ----------------
pa = pd.read_csv(D + "val_rq3_pareto.csv")
w(r"""\subsection{Pareto analysis of the generators}
\label{supp:pareto}
For each $n$ (a common data regime), a generator is Pareto-optimal if no other generator is at least as good on all objectives and strictly better on at least one. Objectives are semantic bigram TV (min), trigram TV (min), near-copy rate (min) and $\Delta$DCR (max), taken at the replicate means of Table~\ref{tab:supp-order2-full}. Table~\ref{tab:supp-pareto} lists the non-dominated generators. On the two-objective projections the LSTM is never on the front; it enters the four-objective front at $n\ge 300$ only as a trade-off point, since no single generator is at least as good on all four objectives (its near-copy rate is lower than that of the light order-2 model, and its trigram TV lower than that of the heavy variant, but the near-copy difference with the light variant is not statistically significant, paired $p\ge 0.06$).
""")
w(r"""\begin{table}[!t]
\caption{Pareto-optimal generators per $n$ (replicate means; M1 = Markov-1, M2L = Markov-2 light, M2H = Markov-2 heavy, L = LSTM).}
\label{tab:supp-pareto}
\centering\small
\begin{tabular}{lccc}
\toprule
$n$ & [bigram TV, $\Delta$DCR] & [trigram TV, $\Delta$DCR] & all four objectives \\
\midrule""")
ab = {"Markov": "M1", "Markov2": "M2L", "Markov2_heavy": "M2H", "LSTM": "L"}
for n in NS:
    g = pa[pa.n == n]
    f = lambda col: ", ".join(ab[m] for m in g[g[col]].model)
    w(f"{n} & {f('pareto_bi_dDCR')} & {f('pareto_tri_dDCR')} & {f('pareto_all4')} \\\\")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- S-D: hypervolume ----------------
hv = pd.read_csv(D + "val_rq3_hypervolume_summary.csv")
w(r"""\subsection{Hypervolume of the generator comparison}
\label{supp:hv}
To summarise each generator by a single number we compute, per paired replicate, the hypervolume dominated by its point in objective space. Within each $n$, every objective is min--max normalised over all (replicate, generator) values, with $\Delta$DCR flipped so that all objectives are minimised, and the reference point is $1.1$ on every axis; we report the dominated volume as a fraction of the ideal box ($1.1^d$), and the exclusive contribution of each generator to the hypervolume of the union of the four generators (zero if dominated by the others). Values are comparable across generators within an $n$, not across $n$. The ranking in Table~\ref{tab:supp-hv} is unchanged when the reference point is moved to $1.5$ or $2.0$, except for a swap between the two order-2 variants at $n{=}50$ with reference $2.0$. The result depends on the objective set: with bigram TV and $\Delta$DCR alone the order-1 model is best, with trigram TV and $\Delta$DCR the light order-2 model is best, and with all four objectives the order-2 variants are best; the LSTM is last under the four-objective set at every $n$ (paired Wilcoxon $p<10^{-4}$ vs.\ both order-2 variants), with an exclusive contribution of at most 0.006.
""")
w(r"""\begin{table}[!t]
\caption{Dominated hypervolume as a fraction of the ideal box (reference point 1.1; mean $\pm$ 95\% CI over 20 paired replicates). Higher is better. Set A: bigram TV and $\Delta$DCR; B: trigram TV and $\Delta$DCR; C: bigram TV, trigram TV, near-copy rate and $\Delta$DCR. Last block: exclusive contribution to the union front (set C).}
\label{tab:supp-hv}
\centering\scriptsize
\begin{tabular}{llccccc}
\toprule
set & generator & $n{=}20$ & $n{=}50$ & $n{=}100$ & $n{=}300$ & $n{=}581$ \\
\midrule""")
sets = [("A", "A [bigramTV, dDCR]", "HV_ind"), ("B", "B [trigramTV, dDCR]", "HV_ind"), ("C", "C [bi, tri, near-copy, dDCR]", "HV_ind"),
        ("C excl.", "C [bi, tri, near-copy, dDCR]", "HV_excl")]
for si, (nm, key, col) in enumerate(sets):
    for i, m in enumerate(MODELS):
        cells = []
        for n in NS:
            r = hv[(hv.set == key) & (hv.n == n) & (hv.model == m)].iloc[0]
            cells.append(f"{r[col + '_norm']:.3f}$\\pm${r[col + '_ci']:.3f}")
        w(f"{nm if i == 0 else ''} & {LAB[m]} & " + " & ".join(cells) + r" \\")
    if si != len(sets) - 1:
        w(r"\midrule")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")
main = []
mw = main.append
mw(r"\begin{table}[!t]")
mw(r"\caption{RQ3 capacity ladder on Paris-OSM (MAT-Sum representation, 20 nested paired replicates; mean $\pm$ 95\% CI). Near-copy: \% of $N{=}200$ synthetic sequences with MUITAS $\ge 0.95$. Best value per $n$ in bold.}")
mw(r"\label{tab:order2}")
mw(r"\centering")
mw(r"\small")
mw(r"\begin{tabular}{llcccc}")
mw(r"\toprule")
mw(r"$n$ & generator & bigram TV $\downarrow$ & trigram TV $\downarrow$ & near-copy \% $\downarrow$ & $\Delta$DCR $\uparrow$ \\")
mw(r"\midrule")
for n in [20, 100, 581]:
    sub = sm[sm.n == n].set_index("model").loc[MODELS]
    best = {"sem_genTV": sub.sem_genTV.min(), "sem_triTV": sub.sem_triTV.min(), "near_pct": sub.near_pct.min(), "dDCR": sub.dDCR.max()}
    for i, m in enumerate(MODELS):
        cells = []
        for c, d in [("sem_genTV", 3), ("sem_triTV", 3), ("near_pct", 2), ("dDCR", 3)]:
            txt = pm(sub.loc[m, c], sub.loc[m, c + "_ci"], d)
            if abs(round(sub.loc[m, c], d) - round(best[c], d)) < 1e-12:
                txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        mw(f"{n if i == 0 else ''} & {LAB[m]} & " + " & ".join(cells) + r" \\")
    if n != 581:
        mw(r"\midrule")
mw(r"\bottomrule")
mw(r"\end{tabular}")
mw(r"\end{table}")
open(os.path.join(OUTDIR, "main_order2_table.tex"), "w").write("\n".join(main))
tex = "\n".join(out)
open(os.path.join(OUTDIR, "supp_additions.tex"), "w").write(tex)
print(len(tex.splitlines()), "lines")
