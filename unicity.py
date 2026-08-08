"""
Unicity analysis (de Montjoye "unique in the crowd", semantic version) -- the uniqueness leg of the
privacy triad. For a record with semantic-label set L, we draw k random labels from L and count how many
records in the dataset contain all k; the record is unique-at-k if exactly one does. unicity_k is the
fraction of unique records (averaged over T random draws), computed via an inverted index.
We report unicity_k (k=1..4) for the REAL dataset (motivation: raw records are highly identifiable) and
for a same-size SYNTHETIC dataset (release does not create a more fingerprint-able artifact).
Datasets handled: Paris (all 581) and GeoLife (2000 sampled sessions). Uses the common OSM-label space.
"""
import os, sys, numpy as np, pandas as pd
from collections import Counter, defaultdict

KS = [1, 2, 3, 4]; T = 25; N_GEO = 2000
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def rle(x):
    out = []
    for v in x:
        if not out or out[-1] != v:
            out.append(v)
    return out


def unicity(label_sets, seed=0):
    rng = np.random.default_rng(seed)
    inv = defaultdict(set)
    for i, s in enumerate(label_sets):
        for l in s:
            inv[l].add(i)
    out = {}
    for k in KS:
        elig = [i for i, s in enumerate(label_sets) if len(s) >= k]
        if not elig:
            out[k] = float("nan"); continue
        uniq = tot = 0
        for i in elig:
            ls = list(label_sets[i])
            for _ in range(T):
                draw = rng.choice(len(ls), k, replace=False)
                match = set.intersection(*[inv[ls[d]] for d in draw])
                uniq += (len(match) == 1); tot += 1
        out[k] = uniq / tot
    return out


def build_vocab_and_synth(state_seqs, label_seqs, n_synth, K=5, seed=0):
    """MAT-Sum min-support vocab on the given records; order-1 Markov; return real & synth label sets."""
    from val_rq3 import markov  # reuse the same generator
    raw = [rle(v) for v in state_seqs]
    fs = Counter(s for v in raw for s in v)
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    fset = set(freq)
    remap = {s: (s if s in fset else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in v]) for v in raw]
    dom = defaultdict(Counter)
    for v, lv in zip(state_seqs, label_seqs):
        for s, l in zip(v, lv):
            dom[remap[s]][l] += 1
    dom = {k2: c.most_common(1)[0][0] for k2, c in dom.items()}
    grng = np.random.default_rng(seed)
    synth = markov(real, n_synth, grng)
    real_lab = [set(lv) for lv in label_seqs]
    synth_lab = [set(dom[s] for s in v) for v in synth]
    return real_lab, synth_lab


def load_paris():
    import matsum_summarize as MS
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    st, lb = [], []
    for t, g in sem.groupby("tid", sort=False):
        st.append(list(g["sig2"])); lb.append(list(g["top1"]))
    return st, lb


def load_geolife(n_sample):
    import pickle
    with open(os.path.join(os.environ["MATSUM_OUT"], "geolife_sess_cache.pkl"), "rb") as f:
        sess, _ = pickle.load(f)
    tids = list(sess.keys())
    idx = np.random.default_rng(0).choice(len(tids), min(n_sample, len(tids)), replace=False)
    st = [sess[tids[i]][0] for i in idx]; lb = [sess[tids[i]][1] for i in idx]
    return st, lb


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
    res = {}
    if which in ("paris", "both"):
        print("[unicity] Paris", flush=True)
        st, lb = load_paris()
        real_lab, synth_lab = build_vocab_and_synth(st, lb, len(st), seed=7)
        res["Paris-real"] = unicity(real_lab, 1); res["Paris-synth"] = unicity(synth_lab, 2)
    if which in ("geolife", "both"):
        print("[unicity] GeoLife", flush=True)
        st, lb = load_geolife(N_GEO)
        real_lab, synth_lab = build_vocab_and_synth(st, lb, len(st), seed=7)
        res["GeoLife-real"] = unicity(real_lab, 1); res["GeoLife-synth"] = unicity(synth_lab, 2)

    df = pd.DataFrame({name: [d[k] for k in KS] for name, d in res.items()}, index=[f"k={k}" for k in KS])
    df.to_csv(os.path.join(OUT, f"unicity_{which}.csv"))
    pd.set_option("display.width", 200)
    print("\n=== Unicity  (fraction of records uniquely identified by k semantic labels) ===")
    print(df.to_string())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sty = {"Paris-real": ("#8856a7", "-", "o"), "Paris-synth": ("#8856a7", "--", "s"),
           "GeoLife-real": ("#2c7fb8", "-", "o"), "GeoLife-synth": ("#2c7fb8", "--", "s")}
    for name, d in res.items():
        c, ls, mk = sty.get(name, ("#555", "-", "o"))
        ax.plot(KS, [d[k] for k in KS], ls, marker=mk, color=c, lw=2, label=name)
    ax.set_xticks(KS); ax.set_xlabel("k semantic labels revealed"); ax.set_ylabel("unicity (fraction uniquely identified)")
    ax.set_title("Unicity: real vs synthetic (solid = real · dashed = synthetic)")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    p = os.path.join(FIG, f"unicity_{which}.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
