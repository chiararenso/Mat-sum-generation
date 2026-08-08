"""
Scaled validation on the Paris-581 MAT dataset (Pugliese et al. 2025).
Runs the (m,k) fidelity/privacy sweep of the synthetic generator with a
VECTORISED MUITAS: symbols come from a finite vocabulary, so a precomputed
S×S subset matrix turns each MUITAS similarity into two numpy reductions,
making nearest-neighbour / DCR tractable over hundreds of real trajectories.

Env: MATSUM_OUT must point at the Paris val output dir.
"""
import os, numpy as np, pandas as pd
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]
N_SYNTH = 200
REF_CAP = 150           # real trajectories sampled for nearest-neighbour / DCR
CONFIGS = [(1, 5), (2, 5), (3, 5), (2, 2), (3, 2)]
rng = np.random.default_rng(0)


def fast_muitas_factory(symbols):
    """R[i,j] = 1 iff symbols[j] ⊆ symbols[i]  (matches the repo's is_subset)."""
    S = len(symbols)
    R = np.zeros((S, S), dtype=bool)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            if b <= a:
                R[i, j] = True

    def sim(ida, idb):
        if len(ida) == 0 or len(idb) == 0:
            return 0.0
        M = R[np.ix_(ida, idb)]
        return (M.max(1).sum() + M.max(0).sum()) / (len(ida) + len(idb))
    return sim


def tv(pa, pb):
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))


def unigram(id_seqs):
    from collections import Counter
    c = Counter(s for v in id_seqs for s in v); n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def bigram(id_seqs):
    from collections import Counter
    c = Counter()
    for v in id_seqs:
        for a, b in zip(v, v[1:]):
            c[(a, b)] += 1
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def main():
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    rows = []
    for m, k in CONFIGS:
        raw = SY.build_visits(sem, m)
        raw_alpha = len({s for v in raw.values() for s, _ in v})
        seqs, support, _, _ = SY.anonymize_vocabulary(raw, k)
        symbols = sorted({s for v in seqs.values() for s, _ in v}, key=lambda z: sorted(z))
        sid = {s: i for i, s in enumerate(symbols)}
        sim = fast_muitas_factory(symbols)

        # real id-sequences
        real_ids = [np.array([sid[s] for s, _ in v]) for v in seqs.values()]

        # fit Markov + sample
        SY.rng = np.random.default_rng(42)
        model = SY.SemanticMarkov(alpha=0.1).fit(seqs)
        synth = [SY.seq_symbols(model.sample(1000)) for _ in range(N_SYNTH)]
        synth_ids = [np.array([sid[s] for s in v]) for v in synth]

        # fidelity
        uni_tv = tv(unigram(real_ids), unigram(synth_ids))
        bi_tv = tv(bigram(real_ids), bigram(synth_ids))

        # nearest-neighbour reference sample
        ref_idx = rng.choice(len(real_ids), min(REF_CAP, len(real_ids)), replace=False)
        ref = [real_ids[i] for i in ref_idx]
        mu = [max(sim(s, r) for r in ref) for s in synth_ids]
        dcr_s = float(1 - np.mean(mu))
        # real->real baseline within the sample (leave-one-out)
        dcr_r = np.mean([1 - max(sim(ref[i], ref[j]) for j in range(len(ref)) if j != i)
                         for i in range(len(ref))])
        # exact copies vs ALL real (cheap tuple compare)
        real_tuples = set(tuple(v.tolist()) for v in real_ids)
        exact = sum(tuple(s.tolist()) in real_tuples for s in synth_ids)

        rows.append({"m": m, "k": k, "raw_alphabet": raw_alpha, "anon_alphabet": len(symbols),
                     "unigram_TV": round(uni_tv, 3), "bigram_TV": round(bi_tv, 3),
                     "MUITAS_to_real": round(float(np.mean(mu)), 3),
                     "DCR_synth": round(dcr_s, 3), "DCR_real_base": round(float(dcr_r), 3),
                     "exact_copies": exact})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "val_sweep_mk.csv"), index=False)
    print("\nsaved val_sweep_mk.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
