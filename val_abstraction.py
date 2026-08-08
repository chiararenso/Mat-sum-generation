"""
The role of the abstraction (Paris-581). Three abstractions, all reduced to ~T states and fed
to the SAME order-1 Markov generator, so any difference is due to HOW the state space is built:
  - Grid-Markov    : ~T spatial grid cells
  - Cluster-Markov : ~T spatial KMeans clusters (a la graph-based approaches)
  - MAT-Sum-Markov : ~T semantic symbols (m=2, k=5 -> 328)

Fairness: every state is also mapped, post-hoc, to the SAME OSM semantic label vocabulary
(each point's dominant top-1 aspect). We then measure how much of the real SEMANTIC structure
survives the abstraction and the generation:
  - generic bigram-TV        : state-level sequence fidelity (each vs its own real)
  - semantic distortion       : TV(real states -> dominant label bigram , ground-truth label bigram)
                                = how much semantic structure the ~T-state abstraction itself loses
  - semantic generation-TV    : TV(synthetic states -> dominant label bigram , ground-truth)
Ground truth = real per-point top-1 label sequences. Lower is better everywhere.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
TARGET, N_SYNTH, SEED = 328, 200, 0
rng = np.random.default_rng(SEED)


def rle(vals):
    out = []
    for v in vals:
        if not out or out[-1] != v:
            out.append(v)
    return out


def tv(pa, pb):
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))


def bigram(seqs):
    c = Counter()
    for v in seqs:
        for a, b in zip(v, v[1:]):
            c[(a, b)] += 1
    n = sum(c.values()) or 1
    return {k: x / n for k, x in c.items()}


def markov_gen(real_seqs, n):
    """order-1 Markov over arbitrary hashable states; length ~ empirical."""
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real_seqs:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = list(states); idx = {s: i for i, s in enumerate(states)}
    lengths = [len(v) for v in real_seqs if len(v) > 0]
    def nxt(state):
        c = np.array([trans[state][s] for s in states], float)
        p = (c + 0.1) / (c.sum() + 0.1 * len(states))
        return states[rng.choice(len(states), p=p)]
    out = []
    for _ in range(n):
        L = int(rng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            st = nxt(st); seq.append(st)
        out.append(seq)
    return out


def main():
    print("[load] Paris-581 + semantic mapping")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    xy = gpd.GeoSeries(sem.geometry, crs="EPSG:4326").to_crs(2154)
    sem["x"] = xy.x.values; sem["y"] = xy.y.values

    # ---- ground-truth semantic label sequences (per-point top-1) ----
    gt_label_seqs = [rle(g["top1"].tolist()) for _, g in sem.groupby("tid", sort=False)]
    gt_bi = bigram(gt_label_seqs)

    # ---- MAT-Sum states (m=2, k=5) ----
    raw = SY.build_visits(sem, 2); seqs, sup, _, _ = SY.anonymize_vocabulary(raw, 5)
    fs = Counter(s for v in raw.values() for s, _ in v)
    frequent = [s for s, c in fs.items() if c >= 5 and len(s) > 0]
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    sem["matsum"] = sem["sig2"].map(lambda s: remap.get(s, s))

    # ---- Grid states: tune cell size to ~TARGET visited cells ----
    lo, hi = 0.0015, 0.06
    for _ in range(30):
        g = (lo + hi) / 2
        nvis = len(set(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int))))
        if nvis > TARGET: lo = g
        else: hi = g
    g = (lo + hi) / 2
    sem["grid"] = list(zip((sem.lat / g).round().astype(int), (sem.lon / g).round().astype(int)))

    # ---- Cluster states: KMeans(TARGET) ----
    XY = sem[["x", "y"]].to_numpy()
    sub = XY[rng.choice(len(XY), min(40000, len(XY)), replace=False)]
    km = KMeans(n_clusters=TARGET, n_init=3, random_state=0).fit(sub)
    sem["cluster"] = km.predict(XY)

    rows = []
    for name, col in [("Grid-Markov", "grid"), ("Cluster-Markov", "cluster"), ("MAT-Sum-Markov", "matsum")]:
        # per-trajectory state sequences (RLE) + state->dominant label
        state_seqs, dom = [], {}
        tmp = sem.groupby(col)["top1"].agg(lambda s: s.value_counts().index[0])
        dom = tmp.to_dict()
        for _, gtr in sem.groupby("tid", sort=False):
            state_seqs.append(rle(gtr[col].tolist()))
        n_states = sem[col].nunique()

        # generic (state-level) fidelity: synth vs own real
        synth = markov_gen(state_seqs, N_SYNTH)
        gen_bi = tv(bigram(synth), bigram(state_seqs))

        # semantic distortion of the abstraction: real states -> dominant label bigram vs GT
        real_lab = [rle([dom[s] for s in seq]) for seq in state_seqs]
        distortion = tv(bigram(real_lab), gt_bi)
        # semantic generation: synth states -> dominant label bigram vs GT
        synth_lab = [rle([dom[s] for s in seq]) for seq in synth]
        gen_sem = tv(bigram(synth_lab), gt_bi)

        # privacy: exact state-sequence copies of real
        real_tuples = set(tuple(s) for s in state_seqs)
        exact = sum(tuple(s) in real_tuples for s in synth)

        rows.append({"method": name, "n_states": n_states,
                     "generic_bigramTV": round(gen_bi, 3),
                     "semantic_distortion": round(distortion, 3),
                     "semantic_genTV": round(gen_sem, 3),
                     "exact_copies": exact, "semantic_states": "yes" if name.startswith("MAT") else "no"})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "val_abstraction.csv"), index=False)
    print("\nground-truth label vocabulary size:", len({l for s in gt_label_seqs for l in s}))
    print(df.to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w/2, df.semantic_distortion, w, color="#8aa0b3", label="abstraction distortion (real→labels)")
    ax.bar(x + w/2, df.semantic_genTV, w, color="#d95f0e", label="generation (synth→labels)")
    for i, v in enumerate(df.generic_bigramTV):
        ax.plot(i, v, "D", color="#2c7fb8", ms=8)
    ax.plot([], [], "D", color="#2c7fb8", label="generic state-bigram TV")
    ax.set_xticks(x); ax.set_xticklabels(df.method)
    ax.set_ylabel("TV distance  (↓ better)")
    ax.set_title(f"Role of the abstraction · Paris-581 · ~{TARGET} states, same Markov")
    ax.legend(fontsize=9)
    fig.tight_layout(); p = os.path.join(FIG, "val_abstraction.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("figure:", p)


if __name__ == "__main__":
    main()
