"""
Three additional evaluation measures for the paper, reusing the SAME operating point
(MAT-Sum m=2, k=5) and the SAME order-1 Markov generator (matsum_synth.SemanticMarkov):

  1. DISTRIBUTION MATCH  - sequence-length and dwell-time distributions, real vs synthetic
                           (Jensen-Shannon divergence + two-sample KS).
  2. DIVERSITY / COVERAGE - generative precision & recall over semantic bigrams, distinct-sequence
                           ratio, and novel-sequence rate (guards against mode collapse / copying).
  3. VOCABULARY k-ANONYMITY - support of each released semantic location, in VISITS and in DISTINCT
                           INDIVIDUALS (the privacy k); min / 5th-pct / median.

Dataset-agnostic: reads the prepared outputs in MATSUM_OUT. R independent generations -> mean +/- 95% CI
for the synthetic-dependent measures. Vocabulary k-anonymity is a property of the real release (single value).
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS
import matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
DS  = os.environ.get("MATSUM_DS", "Paris")     # label for prints/figure
M_TOP, K_SUP, R = 2, 5, 5          # N_synth is set = number of real sequences (matched-size comparison)
try:
    from scipy.stats import t as tdist, ks_2samp
    TCRIT = float(tdist.ppf(0.975, df=R - 1))
except Exception:
    TCRIT, ks_2samp = 2.262, None


def js_divergence(a, b, bins):
    pa, _ = np.histogram(a, bins=bins, density=False)
    pb, _ = np.histogram(b, bins=bins, density=False)
    pa = pa / (pa.sum() or 1); pb = pb / (pb.sum() or 1)
    m = 0.5 * (pa + pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)      # in [0,1] bits


def bigrams(seqs):
    c = Counter()
    for v in seqs:
        s = [x for x, _ in v]
        for a, b in zip(s, s[1:]):
            c[(a, b)] += 1
    return c


def main():
    print(f"[{DS}] load + build MAT-Sum vocabulary (m={M_TOP}, k={K_SUP})", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    raw = SY.build_visits(sem, M_TOP)
    if os.environ.get("GEOLIFE_FILTER") == "1":
        MIN_VISITS, MIN_SESS = 10, 3            # same session filter as the paper's GeoLife E-series
        raw = {t: v for t, v in raw.items() if len(v) >= MIN_VISITS}
        cnt = Counter(t.split("__")[0] for t in raw)
        keep_u = {u for u, c in cnt.items() if c >= MIN_SESS}
        raw = {t: v for t, v in raw.items() if t.split("__")[0] in keep_u}
        print(f"  session filter: {len(raw)} valid sessions from {len(keep_u)} users "
              f"(>= {MIN_VISITS} visits/session, >= {MIN_SESS} sessions/user)", flush=True)
    seqs_d, support, n_freq, merged = SY.anonymize_vocabulary(raw, K_SUP)
    real = list(seqs_d.values())
    real_syms = [[s for s, _ in v] for v in real]
    real_seq_keys = set(tuple(s) for s in real_syms)
    len_r = np.array([len(v) for v in real])
    dw_r = np.array([d for v in real for _, d in v]) / 60.0          # minutes
    B_real = bigrams(real); B_real_set = set(B_real)
    N_SYNTH = len(real)                     # matched-size synthetic release
    print(f"  {len(real)} real sequences | alphabet {len(support)} | min visit-support {min(support.values())} | N_synth={N_SYNTH}", flush=True)

    # ---------- 3. vocabulary k-anonymity (real release, single value) ----------
    indiv = lambda t: str(t).split("__")[0]   # individual = user (GeoLife tid=user__session; Paris tid=individual)
    state_indiv = defaultdict(set)       # state -> set of individuals that ever visit it
    for tid, v in seqs_d.items():
        for s, _ in v:
            state_indiv[s].add(indiv(tid))
    indiv_support = np.array([len(t) for t in state_indiv.values()])
    visit_support = np.array(list(support.values()))
    kanon = dict(min_visits=int(visit_support.min()),
                 min_individuals=int(indiv_support.min()),
                 p5_individuals=float(np.percentile(indiv_support, 5)),
                 median_individuals=float(np.median(indiv_support)),
                 states=len(support))

    # ---------- 1 & 2 over R generations ----------
    acc = defaultdict(list)
    for r in range(R):
        SY.rng = np.random.default_rng(2000 + r)                    # reseed the generator
        model = SY.SemanticMarkov(alpha=0.1).fit(seqs_d)
        synth = [model.sample(SY.MAX_LEN) for _ in range(N_SYNTH)]
        syn_syms = [[s for s, _ in v] for v in synth]
        len_s = np.array([len(v) for v in synth])
        dw_s = np.array([d for v in synth for _, d in v]) / 60.0
        # 1. distribution match
        lmax = int(max(len_r.max(), len_s.max()))
        acc["len_JS"].append(js_divergence(len_r, len_s, bins=np.arange(0, lmax + 5, 2)))
        acc["dwell_JS"].append(js_divergence(np.clip(dw_r, 0, 120), np.clip(dw_s, 0, 120),
                                             bins=np.linspace(0, 120, 31)))
        if ks_2samp is not None:
            acc["len_KS"].append(float(ks_2samp(len_r, len_s).statistic))
            acc["dwell_KS"].append(float(ks_2samp(np.clip(dw_r, 0, 120), np.clip(dw_s, 0, 120)).statistic))
        # 2. diversity / coverage
        B_syn = bigrams(synth); B_syn_set = set(B_syn)
        inter = len(B_real_set & B_syn_set)
        acc["coverage_recall"].append(inter / (len(B_real_set) or 1))     # real bigrams covered
        acc["precision"].append(inter / (len(B_syn_set) or 1))            # synth bigrams that are real
        acc["distinct_seq_ratio"].append(len(set(tuple(s) for s in syn_syms)) / (len(set(tuple(s) for s in real_syms)) or 1))
        acc["novel_seq_rate"].append(np.mean([tuple(s) not in real_seq_keys for s in syn_syms]))
        print(f"  gen {r+1}/{R}", flush=True)

    def agg(k):
        a = np.array(acc[k], float)
        return a.mean(), TCRIT * a.std(ddof=1) / np.sqrt(len(a))

    rows = []
    print(f"\n===== {DS}  (MAT-Sum m={M_TOP},k={K_SUP}; order-1 Markov; N_synth={N_SYNTH}; R={R}) =====")
    print("\n[1] Distribution match  (0 = identical)")
    for k, lab in [("len_JS", "sequence-length JS"), ("dwell_JS", "dwell-time JS"),
                   ("len_KS", "sequence-length KS"), ("dwell_KS", "dwell-time KS")]:
        if acc[k]:
            m, ci = agg(k); print(f"   {lab:22s} {m:.3f} ± {ci:.3f}"); rows.append((DS, k, m, ci))
    print("\n[2] Diversity / coverage")
    for k, lab in [("coverage_recall", "bigram coverage (recall)"), ("precision", "bigram precision"),
                   ("distinct_seq_ratio", "distinct-seq ratio synth/real"), ("novel_seq_rate", "novel-sequence rate")]:
        m, ci = agg(k); print(f"   {lab:30s} {m:.3f} ± {ci:.3f}"); rows.append((DS, k, m, ci))
    print("\n[3] Vocabulary k-anonymity  (real release)")
    print(f"   states={kanon['states']}  min visit-support={kanon['min_visits']}")
    print(f"   distinct INDIVIDUALS per state: min={kanon['min_individuals']}  "
          f"5th-pct={kanon['p5_individuals']:.0f}  median={kanon['median_individuals']:.0f}")
    for k, v in kanon.items():
        rows.append((DS, f"kanon_{k}", v, 0.0))

    pd.DataFrame(rows, columns=["dataset", "metric", "value", "ci95"]).to_csv(
        os.path.join(OUT, "val_extra_metrics.csv"), index=False)

    # ---------- figure (uses the LAST generation for the histograms) ----------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    RC, SC = "#2c7fb8", "#d95f0e"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    lmax = int(max(len_r.max(), len_s.max()))
    bl = np.arange(0, lmax + 5, max(2, lmax // 30))
    ax[0].hist(len_r, bins=bl, density=True, alpha=.6, color=RC, label="real")
    ax[0].hist(len_s, bins=bl, density=True, alpha=.6, color=SC, label="synthetic")
    ax[0].set_title(f"(A) Sequence length  (JS={agg('len_JS')[0]:.3f})"); ax[0].set_xlabel("# semantic visits"); ax[0].legend()
    ax[1].hist(np.clip(dw_r, 0, 120), bins=np.linspace(0, 120, 31), density=True, alpha=.6, color=RC, label="real")
    ax[1].hist(np.clip(dw_s, 0, 120), bins=np.linspace(0, 120, 31), density=True, alpha=.6, color=SC, label="synthetic")
    ax[1].set_title(f"(B) Dwell-time  (JS={agg('dwell_JS')[0]:.3f})"); ax[1].set_xlabel("minutes (clipped 120)"); ax[1].legend()
    cov, _ = agg("coverage_recall"); prec, _ = agg("precision"); nov, _ = agg("novel_seq_rate")
    ax[2].bar(["coverage\n(recall)", "precision", "novel-seq\nrate"], [cov, prec, nov],
              color=["#2c7fb8", "#5BC0BE", "#F4A259"])
    ax[2].set_ylim(0, 1.05); ax[2].set_title("(C) Diversity / coverage")
    for i, val in enumerate([cov, prec, nov]): ax[2].text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=10)
    fig.suptitle(f"Additional measures - {DS}  (MAT-Sum m={M_TOP},k={K_SUP}; order-1 Markov; "
                 f"k-anon: min {kanon['min_individuals']} individuals/state)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, f"fig_extra_metrics_{DS.lower()}.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
