"""
RQ2 -- Data efficiency & memorization (Paris-581, MAT-Sum + order-1 Markov, m=2, k=5).

Nested paired resampling: R replicates; in each, one random permutation of the 581 individuals
yields nested subsets D20 subset D50 subset D100 subset D300 subset D581. FIXED abstraction
parameters: the map and OSM aspects are public and fixed (semantic_mapping), but the k-merge,
transition counts, dwell/length and hence the EFFECTIVE number of states are estimated only on the
sampled users. The DCR real-to-real baseline is recomputed per sample and per replicate.

Per (n, replicate): quality [native bigram-TV (primary), unigram/trigram-TV, semantic generation-TV],
empirical disclosure risk [exact copies / N_SYN, DCR synth->training, DCR real->real baseline],
representation diagnostics [effective states, transition sparsity, median support, #observed bigrams].
Output: 3-panel figure (semantic gen-TV; DCR synth vs baseline; exact copies) with mean +/- 95% CI.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
import matsum_summarize as MS

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K, N_SYN = 2, 5, 200
NS = [20, 50, 100, 300, 581]
R, REF_CAP = 20, 120
try:
    from scipy.stats import t as _t
    TCRIT = float(_t.ppf(0.975, df=R - 1))
except Exception:
    TCRIT = 2.093
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def rle(x):
    out = []
    for v in x:
        if not out or out[-1] != v:
            out.append(v)
    return out


def ngram(seqs, k):
    c = Counter()
    for v in seqs:
        for i in range(len(v) - k + 1):
            c[tuple(v[i:i+k])] += 1
    n = sum(c.values()) or 1
    return {t: x / n for t, x in c.items()}


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def subset_matrix(symbols):
    S = len(symbols); Rm = np.zeros((S, S), bool)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            if b <= a:
                Rm[i, j] = True
    return Rm


def sim(Rm, a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    m = Rm[np.ix_(a, b)]
    return (m.max(1).sum() + m.max(0).sum()) / (len(a) + len(b))


def markov(real, n, grng):
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = list(states); lengths = [len(v) for v in real if len(v)] or [1]
    def nxt(st):
        c = np.array([trans[st][s] for s in states], float)
        return states[grng.choice(len(states), p=(c + 0.1) / (c.sum() + 0.1 * len(states)))]
    out = []
    for _ in range(n):
        L = int(grng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            st = nxt(st); seq.append(st)
        out.append(seq)
    return out


def run_config(tids, per, grng):
    raw = {t: rle(per[t][0]) for t in tids}                      # visit-level sig2 (RLE) per user
    fs = Counter(s for t in tids for s in raw[t])
    frequent = [s for s, c in fs.items() if c >= K and len(s) > 0]
    if not frequent:
        frequent = [fs.most_common(1)[0][0]]
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids]        # real symbol sequences (post k-merge)
    # dominant OSM label per symbol + ground-truth label sequences (per point)
    dom = defaultdict(Counter); gt = []
    for t in tids:
        sig_l, top_l = per[t]
        for s, l in zip(sig_l, top_l):
            dom[remap[s]][l] += 1
        gt.append(rle(top_l))
    dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}

    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = subset_matrix(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]

    synth = markov(real, N_SYN, grng)
    synth_ids = [np.array([sid[s] for s in v]) for v in synth]

    # quality
    nat_bi = tv(ngram(real, 2), ngram(synth, 2))
    nat_uni = tv(ngram(real, 1), ngram(synth, 1))
    nat_tri = tv(ngram(real, 3), ngram(synth, 3))
    synth_lab = [rle([dom[s] for s in v]) for v in synth]
    sem_gen = tv(ngram(synth_lab, 2), ngram(gt, 2))

    # empirical disclosure risk
    real_tup = set(tuple(v) for v in real)
    exact = sum(tuple(v) in real_tup for v in synth)             # out of N_SYN
    ref = [real_ids[i] for i in grng.choice(len(real_ids), min(REF_CAP, len(real_ids)), replace=False)]
    dcr_s = float(1 - np.mean([max(sim(Rm, s, r) for r in ref) for s in synth_ids]))
    dcr_b = float(np.mean([1 - max(sim(Rm, ref[i], ref[j]) for j in range(len(ref)) if j != i)
                           for i in range(len(ref))]))

    # diagnostics
    supp = Counter(s for v in real for s in v)
    obs_bi = set().union(*[{(a, b) for a, b in zip(v, v[1:])} for v in real]) if real else set()
    S = len(symbols)
    return dict(n_states=S, sparsity=round(1 - len(obs_bi) / (S * S) if S else 0, 4),
                median_support=int(np.median(list(supp.values()))) if supp else 0,
                obs_bigrams=len(obs_bi),
                native_biTV=nat_bi, native_uniTV=nat_uni, native_triTV=nat_tri, sem_genTV=sem_gen,
                exact_copies=exact, DCR_synth=dcr_s, DCR_base=dcr_b, dDCR=dcr_s - dcr_b)


def main():
    print("[load] Paris-581 + fixed public map/aspects")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: (list(g["sig2"]), list(g["top1"])) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))
    print(f"  {len(tids)} individuals, {len(sem)} points")

    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(tids)
        for n in NS:
            grng = np.random.default_rng(10_000 * r + n)
            res = run_config(list(perm[:n]), per, grng)
            rows.append({"rep": r, "n": n, **res})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw_df = pd.DataFrame(rows); raw_df.to_csv(os.path.join(OUT, "val_rq2_raw.csv"), index=False)

    # aggregate mean +/- 95% CI per n
    metrics = ["native_biTV", "native_uniTV", "native_triTV", "sem_genTV", "exact_copies",
               "DCR_synth", "DCR_base", "dDCR", "n_states", "sparsity", "median_support", "obs_bigrams"]
    agg = []
    for n in NS:
        sub = raw_df[raw_df.n == n]; row = {"n": n}
        for m in metrics:
            a = sub[m].to_numpy(float)
            row[m] = round(float(a.mean()), 4)
            row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "val_rq2_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== summary (mean +/- 95% CI) ===")
    print(adf[["n", "native_biTV", "sem_genTV", "exact_copies", "DCR_synth", "DCR_base", "dDCR",
               "n_states", "obs_bigrams", "median_support"]].to_string(index=False))

    # ---- 3-panel figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = adf.n.to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    ax[0].errorbar(x, adf.sem_genTV, yerr=adf.sem_genTV_ci, fmt="o-", color="#2c7fb8", capsize=3, lw=2)
    ax[0].set_title("(A) Fidelity"); ax[0].set_ylabel("semantic generation TV (↓)")
    ax[1].errorbar(x, adf.DCR_synth, yerr=adf.DCR_synth_ci, fmt="s-", color="#d95f0e", capsize=3, lw=2, label="synth→training")
    ax[1].errorbar(x, adf.DCR_base, yerr=adf.DCR_base_ci, fmt="o--", color="#555", capsize=3, lw=2, label="real→real baseline")
    ax[1].set_title("(B) Distance to real records"); ax[1].set_ylabel("DCR (↑ safer)"); ax[1].legend(fontsize=9)
    ax[2].errorbar(x, adf.exact_copies, yerr=adf.exact_copies_ci, fmt="D-", color="#6a53a6", capsize=3, lw=2)
    ax[2].set_title("(C) Exact copies"); ax[2].set_ylabel(f"exact copies (of {N_SYN})")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x)
    fig.suptitle(f"RQ2 · MAT-Sum+Markov (m={M},k={K}) · {R} nested paired replicates · fixed abstraction parameters", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "val_rq2.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
