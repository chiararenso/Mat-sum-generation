"""
MAT-SUM applied to the Akkodis (Paris) GPS dataset.
STAGE 4 - Evaluation: summarization rate (S_rate) + semantic quality (MUITAS),
as a function of the similarity threshold tau (the paper's core evaluation).

MUITAS class is the verbatim implementation from chiarap2/MAT-Sum
(evaluation/muitas.py), Petry et al. 2019. GPS config: feature = aspect set,
distance = is_subset, threshold 0, weight 1.

Semantic quality is measured between the ORIGINAL semantic trajectory (one entry
per GPS point, aspect set = the semantic location it maps to) and the SUMMARIZED
trajectory (one entry per representative semantic location). For the summarized
entry we use the UNION of the aspect sets merged into it (the repo aggregates the
aspect/'label' columns by union during summarization2): a summarized location
"matches" an original point when its aspect set is a subset of the point's, so the
score naturally decreases as heavier merging (lower tau) grows those unions.
"""
import os, ast
import numpy as np, pandas as pd, geopandas as gpd
import matsum_summarize as MS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))
TAUS = [0.9, 0.8, 0.7, 0.6, 0.5]


# ---- verbatim MUITAS (Petry et al. 2019, repo evaluation/muitas.py) ----
def is_subset(x, y):
    if x is None or y is None:
        return 1
    return int(not y.issubset(x))


class MUITAS:
    def __init__(self, dist_functions, thresholds, features, weights):
        self.dist_functions = dist_functions
        self.thresholds = thresholds
        self.features = np.array([[f] for f in features])
        self.weights = np.array(weights) / sum(weights)

    def _score(self, p1, p2):
        matches = np.zeros(len(p1))
        for i in range(len(p1)):
            if self.dist_functions[i](p1[i], p2[i]) <= self.thresholds[i]:
                matches[i] = 1
        return matches @ self.weights

    def similarity(self, t1, t2):
        matrix = np.zeros((len(t1), len(t2)))
        for i, p1 in enumerate(t1):
            matrix[i] = [self._score(p1, p2) for p2 in t2]
        parity1 = matrix.max(axis=1).sum()
        parity2 = matrix.max(axis=0).sum()
        return (parity1 + parity2) / (len(t1) + len(t2))


muitas = MUITAS([is_subset], [0], ["label"], [1])


def eval_tau(sem, s1, areas, tau):
    _, similar = MS.build_similarity(areas, tau)
    s2 = MS.summarize2(s1, similar)

    # summarization rates (official definition: distinct categories / raw points)
    npoints = sem.groupby("tid").size()
    n1 = s1.groupby("tid")["category"].nunique()
    n2 = s2.groupby("tid").size()
    rate1 = (1 - n1 / npoints).mean()
    rate2 = (1 - n2 / npoints).mean()

    # MUITAS: original (per point) vs summarized (per representative)
    scores = []
    for tid, g in sem.groupby("tid"):
        t1 = [[a] for a in g["aspects"]]                       # per-point aspect set
        reps = s2[s2.tid == tid]
        t2 = [[a] for a in reps["aspects_union"]]              # per-rep union aspect set
        if len(t1) and len(t2):
            scores.append(muitas.similarity(np.array(t1, dtype=object),
                                            np.array(t2, dtype=object)))
    return rate1, rate2, float(np.mean(scores))


def main():
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    # ensure aspect sets are python sets (semantic_mapping already builds them)
    s1 = MS.summarize1(sem)
    s1["aspects"] = s1["aspects"].apply(lambda x: x if isinstance(x, set) else set(x))

    rows = []
    for tau in TAUS:
        r1, r2, mu = eval_tau(sem, s1, areas, tau)
        rows.append({"tau": tau, "S_rate1": round(r1, 4), "S_rate2": round(r2, 4),
                     "MUITAS": round(mu, 4)})
        print(f"tau={tau}:  S_rate1={r1:.4f}  S_rate2={r2:.4f}  MUITAS={mu:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "evaluation_srate_muitas.csv"), index=False)
    print("\nsaved output/evaluation_srate_muitas.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
