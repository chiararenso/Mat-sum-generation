"""
RQ3: hypervolume view of the generator comparison (order-1 Markov, order-2 Markov light/heavy, LSTM).

For each n and each paired replicate r, every generator is ONE point in objective space. Objectives are
min-max normalised per n over all (replicate, generator) values pooled, so 0 = best observed and 1 =
worst observed; dDCR is flipped (higher is better). The reference point is REF (>1, default 1.1) on every
axis. We report, per replicate then averaged with a 95% CI over R=20:
  HV_ind  hypervolume dominated by the generator's own point (a single-number "quality" score)
  HV_excl exclusive contribution to the hypervolume of the union of the four generators' points
          (HV(all) - HV(all but this one)); 0 means the generator is dominated by the others
Objective sets: A = {bigram TV, dDCR}, B = {trigram TV, dDCR}, C = {bigram TV, trigram TV, near-copy %, dDCR}.
Normalisation is per n, so HV values are comparable across generators within n, not across n.
Sensitivity: the ranking of mean HV_ind is re-computed for REF in {1.1, 1.5, 2.0}.
With <= 4 points the union hypervolume is exact by inclusion-exclusion.
"""
import os, itertools, numpy as np, pandas as pd
from scipy.stats import t as tdist, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MATSUM_OUT", os.path.join(HERE, "data", "paris"))
R = 20
TCRIT = float(tdist.ppf(0.975, R - 1))
MODELS = ["Markov", "Markov2", "Markov2_heavy", "LSTM"]
LABEL = {"Markov": "Markov-1", "Markov2": "Markov-2 (light)", "Markov2_heavy": "Markov-2 (heavy)", "LSTM": "LSTM"}
SETS = {"A [bigramTV, dDCR]": ["sem_genTV", "dDCR"],
        "B [trigramTV, dDCR]": ["sem_triTV", "dDCR"],
        "C [bi, tri, near-copy, dDCR]": ["sem_genTV", "sem_triTV", "near_pct", "dDCR"]}
MAXIMISE = {"dDCR"}


def hv_union(P, ref):
    """Exact hypervolume of the union of boxes [p, ref] (minimisation), P: (m, d), m small."""
    m = len(P); tot = 0.0
    for k in range(1, m + 1):
        for S in itertools.combinations(range(m), k):
            corner = P[list(S)].max(0)
            vol = float(np.prod(np.clip(ref - corner, 0, None)))
            tot += (-1) ** (k + 1) * vol
    return tot


def main():
    here = os.path.join(HERE, "results", "paris", "val_rq3_raw.csv")
    base = pd.read_csv(os.environ.get("RQ3_RAW", here))
    new = pd.read_csv(os.path.join(OUT, "val_rq3_order2_raw.csv"))
    allr = pd.concat([base[base.model.isin(["Markov", "LSTM"])], new], ignore_index=True)
    NS = sorted(allr.n.unique())
    rows = []
    for setname, cols in SETS.items():
        for n in NS:
            sub = allr[allr.n == n]
            X = sub[cols].to_numpy(float).copy()
            for j, c in enumerate(cols):
                if c in MAXIMISE:
                    X[:, j] = -X[:, j]
            lo, hi = X.min(0), X.max(0)
            for ref_v in (1.1, 1.5, 2.0):
                ref = np.full(len(cols), ref_v)
                for r in range(R):
                    pts = {}
                    for m in MODELS:
                        row = sub[(sub.rep == r) & (sub.model == m)]
                        v = row[cols].to_numpy(float)[0].copy()
                        for j, c in enumerate(cols):
                            if c in MAXIMISE:
                                v[j] = -v[j]
                        pts[m] = (v - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
                    P = np.array([pts[m] for m in MODELS])
                    hv_all = hv_union(P, ref)
                    for i, m in enumerate(MODELS):
                        hv_ind = float(np.prod(np.clip(ref - P[i], 0, None)))
                        hv_ex = hv_all - hv_union(np.delete(P, i, axis=0), ref)
                        rows.append({"set": setname, "n": n, "ref": ref_v, "rep": r, "model": m,
                                     "HV_ind": hv_ind, "HV_excl": hv_ex, "HV_union": hv_all,
                                     "HV_max": float(ref_v ** len(cols))})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "val_rq3_hypervolume_raw.csv"), index=False)

    def mci(a):
        a = np.asarray(a, float)
        return a.mean(), TCRIT * a.std(ddof=1) / np.sqrt(len(a))

    main_ref = df[df.ref == 1.1]
    summ = []
    for (setname, n, m), g in main_ref.groupby(["set", "n", "model"]):
        hm, hc = mci(g.HV_ind / g.HV_max); em, ec = mci(g.HV_excl / g.HV_max)
        summ.append({"set": setname, "n": n, "model": m, "HV_ind_norm": hm, "HV_ind_ci": hc,
                     "HV_excl_norm": em, "HV_excl_ci": ec})
    S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "val_rq3_hypervolume_summary.csv"), index=False)

    pd.set_option("display.width", 220)
    for setname in SETS:
        print(f"\n=== set {setname} | REF=1.1 | HV as fraction of the ideal box, mean ± 95% CI over R=20 ===")
        t = S[S.set == setname].copy()
        t["HV_ind"] = t.HV_ind_norm.map("{:.3f}".format) + "±" + t.HV_ind_ci.map("{:.3f}".format)
        t["HV_excl"] = t.HV_excl_norm.map("{:.3f}".format) + "±" + t.HV_excl_ci.map("{:.3f}".format)
        t["model"] = t.model.map(LABEL)
        print(t.pivot(index="model", columns="n", values="HV_ind").reindex([LABEL[m] for m in MODELS]).to_string())
        if setname.startswith("C"):
            print("  -- exclusive contribution to the union front --")
            print(t.pivot(index="model", columns="n", values="HV_excl").reindex([LABEL[m] for m in MODELS]).to_string())

    print("\n=== paired Wilcoxon on HV_ind (REF=1.1), set C: Markov-2 variants vs LSTM and vs Markov-1 ===")
    for setname in ("B [trigramTV, dDCR]", "C [bi, tri, near-copy, dDCR]"):
        print(f"  set {setname}")
        for n in NS:
            g = main_ref[(main_ref.set == setname) & (main_ref.n == n)]
            piv = g.pivot(index="rep", columns="model", values="HV_ind")
            out = []
            for a, b in (("Markov2", "LSTM"), ("Markov2_heavy", "LSTM"), ("Markov", "LSTM"), ("Markov2", "Markov"), ("Markov2_heavy", "Markov")):
                d = (piv[a] - piv[b]).to_numpy()
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                out.append(f"{LABEL[a]}-{LABEL[b]}: {d.mean():+.3f} (p={p:.0e})")
            print(f"    n={n:3d}  " + " | ".join(out))

    print("\n=== ranking of mean HV_ind (best first) under three reference points, set C ===")
    for n in NS:
        line = []
        for ref_v in (1.1, 1.5, 2.0):
            g = df[(df.set.str.startswith("C")) & (df.n == n) & (df.ref == ref_v)].groupby("model").HV_ind.mean()
            line.append(f"ref={ref_v}: " + " > ".join(LABEL[m] for m in g.sort_values(ascending=False).index))
        print(f"  n={n:3d}  " + " || ".join(line))
    print("\nsaved:", os.path.join(OUT, "val_rq3_hypervolume_{raw,summary}.csv"))


if __name__ == "__main__":
    main()
