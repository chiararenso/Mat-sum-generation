"""
MAT-SUM -> synthetic MAT dataset.  PROTOTYPE: Phase 0 + Phase 1a.

Phase 0  - build ordered semantic VISIT sequences (with dwell-times) and an
           anonymized vocabulary: symbol = top-m TF-IDF aspects of a semantic
           location; enforce k-anonymity by merging rare symbols into the
           most Jaccard-similar frequent one.
Phase 1a - mechanistic generator: order-1 semantic Markov chain (with START/END
           and Laplace smoothing) + empirical per-symbol dwell-time; sample N
           synthetic trajectories.
Evaluation - fidelity (unigram/bigram/dwell/length distances, MUITAS synth<->real)
             and privacy (distance-to-closest-record, exact copies, novelty).

Run after matsum_prepare.py + matsum_summarize.py.
"""
import os, json, math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import matsum_summarize as MS
from matsum_muitas import MUITAS, is_subset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))
FIG  = os.environ.get("MATSUM_FIG", os.path.join(HERE, "figures"))
os.makedirs(FIG, exist_ok=True)

# ---- operating point (validated on Paris-581; was m=3,k=2 tuned for the 20-trajectory PoC) ----
M_TOP     = 2      # semantic granularity: top-m TF-IDF aspects define a symbol
K_SUPPORT = 5      # k-anonymity: every symbol must occur >= k times
N_SYNTH   = 2000   # size of the synthetic dataset (was 200, tuned for the 20-trajectory PoC;
                    # too small at n=20000 real for a stable bigram-TV estimate)
ALPHA     = 0.1    # Laplace smoothing for the transition matrix
MAX_LEN   = 120    # safety cap on generated sequence length
SEED      = 42
rng = np.random.default_rng(SEED)


# ============================ Phase 0 ============================
def signature(tfidf, m):
    terms = list(tfidf)[:m] if hasattr(tfidf, "__len__") else []
    return frozenset(terms)


def build_visits(sem, m):
    """Per trajectory, RLE-collapse consecutive equal signatures into visits
    (symbol, dwell). sem is already sorted by (tid, time). Vectorised access
    (to_numpy + zip) so it scales to millions of points."""
    seqs = {}
    for tid, g in sem.groupby("tid", sort=False):
        tfidfs = g["label_tfidf"].to_numpy()
        durs = g["duration"].to_numpy()
        visits = []
        for tf, d in zip(tfidfs, durs):
            sig = signature(tf, m)
            dwell = float(d) if d == d else 0.0            # d==d is False for NaN
            if visits and visits[-1][0] == sig:
                visits[-1][1] += dwell
            else:
                visits.append([sig, dwell])
        seqs[tid] = visits
    return seqs


def anonymize_vocabulary(seqs, k):
    """Merge signatures with support < k into the most Jaccard-similar frequent
    signature (semantic k-anonymity on the vocabulary)."""
    freq = Counter(sig for v in seqs.values() for sig, _ in v)
    frequent = [s for s, c in freq.items() if c >= k and len(s) > 0]
    rare = [s for s in freq if s not in frequent]
    if not frequent:                       # degenerate fallback
        frequent = [freq.most_common(1)[0][0]]
        rare = [s for s in freq if s not in frequent]

    def jaccard(a, b):
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    remap = {s: s for s in frequent}
    for r in rare:
        target = max(frequent, key=lambda s: jaccard(r, s))
        remap[r] = target

    new_seqs, merged = {}, 0
    for tid, v in seqs.items():
        nv = []
        for sig, dwell in v:
            ns = remap[sig]
            if sig != ns:
                merged += 1
            if nv and nv[-1][0] == ns:     # re-collapse if merge made neighbours equal
                nv[-1][1] += dwell
            else:
                nv.append([ns, dwell])
        new_seqs[tid] = nv
    final_support = Counter(sig for v in new_seqs.values() for sig, _ in v)
    return new_seqs, final_support, len(frequent), merged


# ============================ Phase 1a ============================
class SemanticMarkov:
    """Order-1 semantic Markov chain. Length is drawn from the empirical
    trajectory-length distribution (DITRAS-style decoupling of 'how many' from
    'where'), which reproduces realistic lengths without an END state whose
    probability would be diluted across a large symbol alphabet."""
    START = "<START>"

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, seqs):
        self.symbols = sorted({sig for v in seqs.values() for sig, _ in v}, key=lambda s: sorted(s))
        self.idx = {s: i for i, s in enumerate(self.symbols)}
        trans = defaultdict(Counter)
        dwell = defaultdict(list)
        for v in seqs.values():
            prev = self.START
            for sig, d in v:
                trans[prev][sig] += 1
                dwell[sig].append(d)
                prev = sig
        self.trans, self.dwell = trans, dwell
        self.lengths = [len(v) for v in seqs.values()]
        return self

    def _next(self, state):
        counts = np.array([self.trans[state][s] for s in self.symbols], dtype=float)
        probs = (counts + self.alpha) / (counts.sum() + self.alpha * len(self.symbols))
        return self.symbols[rng.choice(len(self.symbols), p=probs)]

    def sample(self, max_len=120):
        L = min(int(rng.choice(self.lengths)), max_len)
        seq, state = [], self.START
        for _ in range(max(L, 1)):
            nxt = self._next(state)
            d = float(rng.choice(self.dwell[nxt])) if self.dwell[nxt] else 0.0
            seq.append([nxt, d])
            state = nxt
        return seq


# ============================ Evaluation ============================
def tv_distance(pa, pb):
    keys = set(pa) | set(pb)
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in keys)


def dist_unigram(seqs):
    c = Counter(sig for v in seqs.values() for sig, _ in v)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def dist_bigram(seqs):
    c = Counter()
    for v in seqs.values():
        syms = [s for s, _ in v]
        for a, b in zip(syms, syms[1:]):
            c[(a, b)] += 1
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


muitas = MUITAS([is_subset], [0], ["label"], [1])


def seq_symbols(v):
    return [sig for sig, _ in v]


def muitas_sim(a, b):
    t1 = np.array([[s] for s in seq_symbols(a)], dtype=object)
    t2 = np.array([[s] for s in seq_symbols(b)], dtype=object)
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    return muitas.similarity(t1, t2)


def dcr(query_seqs, ref_seqs):
    """distance-to-closest-record: 1 - max MUITAS to any reference sequence."""
    out = []
    for q in query_seqs:
        best = max((muitas_sim(q, r) for r in ref_seqs), default=0.0)
        out.append(1 - best)
    return np.array(out)


def main():
    print(f"[Phase 0] build vocabulary  (m={M_TOP}, k={K_SUPPORT})")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    raw = build_visits(sem, M_TOP)
    raw_alpha = len({sig for v in raw.values() for sig, _ in v})
    seqs, support, n_freq, merged = anonymize_vocabulary(raw, K_SUPPORT)
    print(f"  raw symbols: {raw_alpha}  ->  anonymized alphabet: {len(support)} "
          f"(all support >= {min(support.values())}); {merged} visits remapped")
    print(f"  visit-sequence length: mean={np.mean([len(v) for v in seqs.values()]):.1f} "
          f"min={min(len(v) for v in seqs.values())} max={max(len(v) for v in seqs.values())}")

    print("[Phase 1a] fit semantic Markov + dwell, sample synthetic set")
    model = SemanticMarkov(alpha=ALPHA).fit(seqs)
    synth = {f"synth_{i:04d}": model.sample(MAX_LEN) for i in range(N_SYNTH)}
    print(f"  generated {len(synth)} synthetic trajectories "
          f"(real: {len(seqs)});  synth length mean="
          f"{np.mean([len(v) for v in synth.values()]):.1f}")

    print("\n[Fidelity]")
    du_r, du_s = dist_unigram(seqs), dist_unigram(synth)
    db_r, db_s = dist_bigram(seqs), dist_bigram(synth)
    print(f"  unigram TV distance : {tv_distance(du_r, du_s):.4f}   (0 = identical)")
    print(f"  bigram  TV distance : {tv_distance(db_r, db_s):.4f}")
    len_r = np.array([len(v) for v in seqs.values()])
    len_s = np.array([len(v) for v in synth.values()])
    print(f"  length  mean real={len_r.mean():.1f}  synth={len_s.mean():.1f}")
    dw_r = np.array([d for v in seqs.values() for _, d in v]) / 60
    dw_s = np.array([d for v in synth.values() for _, d in v]) / 60
    print(f"  dwell(min) median real={np.median(dw_r):.1f}  synth={np.median(dw_s):.1f}")
    real_list = list(seqs.values())
    # All-pairs MUITAS is O(n^2) and only tractable for small n (fine at n=20, infeasible at
    # n~20000 -- ~4e8 comparisons). Above REF_CAP, use a fixed-seed random subsample as the
    # comparison pool: a quick, honest approximation for a sanity-check run, not a substitute
    # for a properly vectorized full-population audit (see val_paris_eval.py's approach) before
    # drawing final privacy conclusions on the full dataset.
    REF_CAP = 1000
    if len(real_list) > REF_CAP:
        idx = rng.choice(len(real_list), REF_CAP, replace=False)
        ref_list = [real_list[i] for i in idx]
        print(f"  [note] {len(real_list)} real sequences -> sampling {REF_CAP} as the comparison "
              f"pool for MUITAS/DCR (all-pairs is infeasible at this scale)")
    else:
        ref_list = real_list
    n_synth_seqs = len(synth)
    step = max(1, n_synth_seqs // 10)
    mu_best = []
    for i, s in enumerate(synth.values()):
        mu_best.append(max((muitas_sim(s, r) for r in ref_list), default=0.0))
        if (i + 1) % step == 0 or i + 1 == n_synth_seqs:
            print(f"  [progress] synth->real nearest match: {i+1}/{n_synth_seqs}", flush=True)
    print(f"  MUITAS synth->nearest real: mean={np.mean(mu_best):.3f} "
          f"(high = semantically realistic)")

    print("\n[Privacy]")
    dcr_synth = 1 - np.array(mu_best)
    # baseline: real-to-real leave-one-out closest, over the same comparison pool
    n_ref = len(ref_list)
    step2 = max(1, n_ref // 10)
    dcr_real = []
    for i, r in enumerate(ref_list):
        others = ref_list[:i] + ref_list[i+1:]
        dcr_real.append(1 - max((muitas_sim(r, o) for o in others), default=0.0))
        if (i + 1) % step2 == 0 or i + 1 == n_ref:
            print(f"  [progress] real-to-real baseline: {i+1}/{n_ref}", flush=True)
    dcr_real = np.array(dcr_real)
    exact = sum(any(seq_symbols(s) == seq_symbols(r) for r in ref_list) for s in synth.values())
    real_bigrams = set(db_r)
    synth_bigrams = set(db_s)
    novelty = 1 - len(synth_bigrams & real_bigrams) / (len(synth_bigrams) or 1)
    print(f"  DCR synth->real : mean={dcr_synth.mean():.3f}  min={dcr_synth.min():.3f}")
    print(f"  DCR real->real  : mean={dcr_real.mean():.3f}  (baseline)")
    print(f"  exact-copy sequences: {exact}/{len(synth)}  (0 = no verbatim leakage)")
    print(f"  novel bigrams in synth: {novelty*100:.1f}%  (generalization, not memorization)")

    # persist
    def dump(seqs_dict):
        return [{"tid": tid,
                 "symbols": [sorted(s) for s, _ in v],
                 "dwell_s": [round(d, 1) for _, d in v]}
                for tid, v in seqs_dict.items()]
    with open(os.path.join(OUT, "synthetic_semantic.json"), "w") as f:
        json.dump(dump(synth), f, ensure_ascii=False)
    with open(os.path.join(OUT, "real_semantic_sequences.json"), "w") as f:
        json.dump(dump(seqs), f, ensure_ascii=False)
    pd.DataFrame({
        "metric": ["unigram_TV", "bigram_TV", "len_mean_real", "len_mean_synth",
                   "MUITAS_synth_to_real", "DCR_synth_mean", "DCR_real_mean",
                   "exact_copies", "novel_bigrams_pct"],
        "value": [round(tv_distance(du_r, du_s), 4), round(tv_distance(db_r, db_s), 4),
                  round(len_r.mean(), 1), round(len_s.mean(), 1),
                  round(float(np.mean(mu_best)), 3), round(float(dcr_synth.mean()), 3),
                  round(float(dcr_real.mean()), 3), exact, round(novelty*100, 1)],
    }).to_csv(os.path.join(OUT, "synth_eval.csv"), index=False)

    _figures(du_r, du_s, dw_r, dw_s, dcr_synth, dcr_real, len_r, len_s)
    print("\n[done] outputs in output/ (synthetic_semantic.json, synth_eval.csv), figures/")


def _figures(du_r, du_s, dw_r, dw_s, dcr_synth, dcr_real, len_r, len_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # unigram real vs synth (top symbols)
    top = [k for k, _ in Counter(du_r).most_common(12)]
    lab = [",".join(sorted(s))[:22] for s in top]
    x = np.arange(len(top))
    ax[0, 0].bar(x - 0.2, [du_r.get(s, 0) for s in top], 0.4, label="real", color="#2c7fb8")
    ax[0, 0].bar(x + 0.2, [du_s.get(s, 0) for s in top], 0.4, label="synthetic", color="#d95f0e")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(lab, rotation=45, ha="right", fontsize=7)
    ax[0, 0].set_title("Semantic-symbol frequency (top 12)"); ax[0, 0].legend()

    ax[0, 1].hist(np.clip(dw_r, 0, 120), bins=25, density=True, alpha=.6, label="real", color="#2c7fb8")
    ax[0, 1].hist(np.clip(dw_s, 0, 120), bins=25, density=True, alpha=.6, label="synthetic", color="#d95f0e")
    ax[0, 1].set_title("Dwell-time (min, clipped 120)"); ax[0, 1].legend(); ax[0, 1].set_xlabel("min")

    ax[1, 0].hist(len_r, bins=range(0, max(len_r.max(), len_s.max()) + 5, 5), density=True,
                  alpha=.6, label="real", color="#2c7fb8")
    ax[1, 0].hist(len_s, bins=range(0, max(len_r.max(), len_s.max()) + 5, 5), density=True,
                  alpha=.6, label="synthetic", color="#d95f0e")
    ax[1, 0].set_title("Trajectory length (# visits)"); ax[1, 0].legend()

    ax[1, 1].hist(dcr_synth, bins=20, density=True, alpha=.6, label="synth→real", color="#d95f0e")
    ax[1, 1].hist(dcr_real, bins=20, density=True, alpha=.6, label="real→real (baseline)", color="#2c7fb8")
    ax[1, 1].axvline(0, color="k", lw=1, ls=":")
    ax[1, 1].set_title("Distance-to-closest-record  (0 = copy)"); ax[1, 1].legend()
    ax[1, 1].set_xlabel("1 − MUITAS to nearest")

    fig.suptitle(f"Synthetic vs real semantic trajectories  (m={M_TOP}, k={K_SUPPORT}, "
                 f"N={N_SYNTH})", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIG, "synth_vs_real.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  figure:", p)


if __name__ == "__main__":
    main()
