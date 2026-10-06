"""
RQ3 capacity ladder: add an order-2 Markov generator between the order-1 Markov and the LSTM.

Same MAT-Sum representation (m=2, k=5), same nested paired user samples (permutation seed = replicate),
same reference sets / matched real-to-real DCR baseline, same N=200 synthetic sequences per run as
val_rq3.py. The order-1 Markov and LSTM rows are taken from val_rq3_raw.csv (identical samples), so
all comparisons are paired by (replicate, n). We only run the new generator(s).

Order-2 Markov: P(j | a, b) = (C_abj + lam * P1(j | b)) / (C_ab + lam), a Dirichlet-style back-off to the
paper's order-1 model P1(j|b) = (C_bj + alpha) / (C_b + alpha*|V|), alpha = 0.1. Unseen contexts fall
back exactly to P1. Two back-off strengths:
  Markov2        lam = 1            (light back-off: observed contexts dominate)
  Markov2_heavy  lam = alpha*|V|    (same total pseudo-count mass as the order-1 smoothing)
An order-2 chain is fitted on trigram statistics, so semantic trigram TV is no longer "unfitted";
the question is how much of the LSTM's trigram gain and near-copy cost a memory-2 count model reproduces.
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS
from val_rq3 import (rle, ngram, tv, subset_matrix, sim, jac, K, N_SYN, MAXLEN, NS, R, REF_CAP, NEAR, TCRIT)

OUT = os.environ["MATSUM_OUT"]
ALPHA = 0.1
VARIANTS = [("Markov2", None), ("Markov2_heavy", "heavy")]   # lam=1 ; lam=alpha*|V|


def markov2(real, n, grng, lam_mode):
    START = None
    states = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    V = len(states); idx = {s: i for i, s in enumerate(states)}
    lam = 1.0 if lam_mode is None else ALPHA * V
    C1 = np.zeros((V + 1, V)); ctx = defaultdict(dict)
    for v in real:
        a = b = V
        for s in v:
            j = idx[s]; C1[b, j] += 1
            ctx[(a, b)][j] = ctx[(a, b)].get(j, 0) + 1
            a, b = b, j
    P1 = (C1 + ALPHA) / (C1.sum(1, keepdims=True) + ALPHA * V)
    lengths = [len(v) for v in real if len(v)] or [1]
    out = []
    for _ in range(n):
        L = int(grng.choice(lengths)); a = b = V; seq = []
        for _ in range(L):
            p = P1[b] * lam
            c = ctx.get((a, b))
            if c:
                for j, x in c.items():
                    p[j] += x
            p = p / p.sum()
            j = int(grng.choice(V, p=p)); seq.append(states[j]); a, b = b, j
        out.append(seq)
    return out


def run_config(tids, per, grng, rep, n_users):
    raw = {t: rle(per[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    frequent = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]])[:MAXLEN] for t in tids]
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
    real_tup = set(tuple(v) for v in real)
    real_bi, gt_bi = ngram(real, 2), ngram(gt, 2)
    gt_tri = ngram(gt, 3)
    ref = [real_ids[i] for i in grng.choice(len(real_ids), min(REF_CAP, len(real_ids)), replace=False)]
    dcr_base = float(np.mean([1 - max(sim(Rm, ref[i], ref[j]) for j in range(len(ref)) if j != i)
                              for i in range(len(ref))]))

    def evaluate(synth):
        synth_ids = [np.array([sid[s] for s in v]) for v in synth]
        max_all = [max((sim(Rm, s, r) for r in real_ids), default=0.0) for s in synth_ids]
        max_ref = [max((sim(Rm, s, r) for r in ref), default=0.0) for s in synth_ids]
        near = 100.0 * np.mean([m >= NEAR for m in max_all])
        dcr_s = float(1 - np.mean(max_ref))
        synth_lab = [rle([dom[s] for s in v]) for v in synth]
        return {"sem_genTV": tv(ngram(synth_lab, 2), gt_bi), "sem_triTV": tv(ngram(synth_lab, 3), gt_tri),
                "native_biTV": tv(ngram(synth, 2), real_bi),
                "near_pct": float(near), "DCR_synth": dcr_s, "DCR_base": dcr_base, "dDCR": dcr_s - dcr_base,
                "exact_copies": sum(tuple(v) in real_tup for v in synth)}

    out = {}
    for k, (name, mode) in enumerate(VARIANTS):
        mrng = np.random.default_rng(20_000 * rep + n_users + 7 * k)     # separate stream: grng draws unchanged
        out[name] = evaluate(markov2(real, N_SYN, mrng, mode))
    return out, dcr_base


def main():
    print("[load] Paris-581 (fixed public map/aspects)")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: (list(g["sig2"]), list(g["top1"])) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))

    base = pd.read_csv(os.environ.get("RQ3_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               "results", "paris", "val_rq3_raw.csv")))
    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(tids)     # SAME samples as RQ2/RQ3
        for n in NS:
            grng = np.random.default_rng(10_000 * r + n)       # same draws as val_rq3 up to the reference set
            res, dcr_base = run_config(list(perm[:n]), per, grng, r, n)
            ref_dcr = float(base[(base.rep == r) & (base.n == n) & (base.model == "Markov")]["DCR_base"].iloc[0])
            assert abs(ref_dcr - dcr_base) < 1e-9, f"reference set mismatch at rep={r} n={n}: {ref_dcr} vs {dcr_base}"
            for name, _ in VARIANTS:
                rows.append({"rep": r, "n": n, "model": name, **res[name]})
            print(f"  replicate {r+1}/{R}  n={n} done", flush=True)
    new = pd.DataFrame(rows); new.to_csv(os.path.join(OUT, "val_rq3_order2_raw.csv"), index=False)

    allr = pd.concat([base[base.model.isin(["Markov", "LSTM"])], new], ignore_index=True)
    metrics = ["sem_genTV", "sem_triTV", "near_pct", "dDCR", "exact_copies"]
    agg = []
    order = ["Markov", "Markov2", "Markov2_heavy", "LSTM"]
    for n in NS:
        for model in order:
            sub = allr[(allr.n == n) & (allr.model == model)]
            row = {"n": n, "model": model}
            for m in metrics:
                a = sub[m].to_numpy(float)
                row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
            agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "val_rq3_order2_summary.csv"), index=False)
    pd.set_option("display.width", 240)
    print("\n=== capacity ladder (mean +/- 95% CI, R=20 paired replicates) ===")
    show = adf.copy()
    for m in metrics:
        show[m] = show[m].map("{:.3f}".format) + "±" + show[m + "_ci"].map("{:.3f}".format)
    print(show[["n", "model"] + metrics].to_string(index=False))

    from scipy.stats import wilcoxon
    print("\n=== paired differences  d = X_(Markov2 variant) - X_comparator   (TV/near-copy: <0 better for model2; dDCR: >0 better) ===")
    prows = []
    for name in ("Markov2", "Markov2_heavy"):
        for comp in ("Markov", "LSTM"):
            for n in NS:
                a = allr[(allr.n == n) & (allr.model == name)].sort_values("rep")
                b = allr[(allr.n == n) & (allr.model == comp)].sort_values("rep")
                for m in ("sem_genTV", "sem_triTV", "near_pct", "dDCR"):
                    d = a[m].to_numpy(float) - b[m].to_numpy(float)
                    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
                    wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                    prows.append({"model": name, "vs": comp, "n": n, "metric": m, "d_mean": round(float(d.mean()), 4),
                                  "d_z": round(float(dz), 2), "wilcoxon_p": wp})
    pdf = pd.DataFrame(prows); pdf.to_csv(os.path.join(OUT, "val_rq3_order2_paired.csv"), index=False)
    for (name, comp), g in pdf.groupby(["model", "vs"]):
        print(f"\n  {name} - {comp}")
        print(g.pivot(index="n", columns="metric", values="d_mean").to_string())
    print("\nsaved:", os.path.join(OUT, "val_rq3_order2_{raw,summary,paired}.csv"))


if __name__ == "__main__":
    main()
