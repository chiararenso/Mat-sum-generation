"""Sweep semantic granularity (m) and anonymity (k): shows how alphabet size,
fidelity (bigram TV, MUITAS) and privacy (DCR, exact copies) trade off.
This makes concrete the sparsity/fidelity/privacy tension of the prototype."""
import os
import numpy as np, pandas as pd
import matsum_synth as S, matsum_summarize as MS

OUT = os.environ.get("MATSUM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
N = 100

gdf, areas = MS.load()
sem = MS.semantic_mapping(gdf, areas)
real_len_note = None
rows = []
for m in [1, 2, 3]:
    raw = S.build_visits(sem, m)
    raw_alpha = len({sig for v in raw.values() for sig, _ in v})
    for k in [2, 5]:
        S.rng = np.random.default_rng(42)                # reproducible per config
        seqs, support, _, merged = S.anonymize_vocabulary(raw, k)
        model = S.SemanticMarkov(alpha=0.1).fit(seqs)
        synth = {f"s{i}": model.sample(120) for i in range(N)}
        du_r, du_s = S.dist_unigram(seqs), S.dist_unigram(synth)
        db_r, db_s = S.dist_bigram(seqs), S.dist_bigram(synth)
        real_list = list(seqs.values())
        mu = [max((S.muitas_sim(s, r) for r in real_list), default=0.0) for s in synth.values()]
        exact = sum(any(S.seq_symbols(s) == S.seq_symbols(r) for r in real_list)
                    for s in synth.values())
        rows.append({
            "m": m, "k": k, "raw_alphabet": raw_alpha, "anon_alphabet": len(support),
            "unigram_TV": round(S.tv_distance(du_r, du_s), 3),
            "bigram_TV": round(S.tv_distance(db_r, db_s), 3),
            "MUITAS_synth->real": round(float(np.mean(mu)), 3),
            "DCR_synth_mean": round(float(1 - np.mean(mu)), 3),
            "exact_copies": exact,
        })
        print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "synth_sweep_mk.csv"), index=False)
print("\nsaved output/synth_sweep_mk.csv")
print(df.to_string(index=False))
