"""
EXP B - data-scaling on Paris-581 (tests H3, data efficiency).
Fix the semantic vocabulary (from the full city) and the granularity (m,k);
vary the number of TRAINING trajectories n; measure how transition fidelity
and memorisation change with n. Isolates the effect of n from the city size.

Prediction: bigram-TV (vs the full real distribution) falls as n grows, while
the mechanistic Markov keeps DCR >= the real->real baseline (no memorisation)
at every n — including tiny n where a deep model would copy.
"""
import os, numpy as np, pandas as pd
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M_TOP, K = 2, 5                 # fixed operating point (usable region at Paris scale)
NS = [20, 50, 100, 300, 581]
SEEDS = [0, 1, 2]
N_SYNTH = 200
REF_CAP = 150
rng0 = np.random.default_rng(123)


def subset_matrix(symbols):
    S = len(symbols); R = np.zeros((S, S), bool)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            if b <= a:
                R[i, j] = True
    return R


def sim(R, ida, idb):
    if len(ida) == 0 or len(idb) == 0:
        return 0.0
    M = R[np.ix_(ida, idb)]
    return (M.max(1).sum() + M.max(0).sum()) / (len(ida) + len(idb))


def tv(pa, pb):
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))


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
    raw = SY.build_visits(sem, M_TOP)
    seqs_full, support, _, _ = SY.anonymize_vocabulary(raw, K)     # fixed vocabulary
    symbols = sorted({s for v in seqs_full.values() for s, _ in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}
    R = subset_matrix(symbols)
    all_seqs = list(seqs_full.values())
    real_ids_full = [np.array([sid[s] for s, _ in v]) for v in all_seqs]
    bi_full = bigram(real_ids_full)
    ref_idx = rng0.choice(len(real_ids_full), min(REF_CAP, len(real_ids_full)), replace=False)
    ref = [real_ids_full[i] for i in ref_idx]
    dcr_base = np.mean([1 - max(sim(R, ref[i], ref[j]) for j in range(len(ref)) if j != i)
                        for i in range(len(ref))])

    rows = []
    for n in NS:
        bt, dcr, exact = [], [], []
        for seed in SEEDS:
            r = np.random.default_rng(seed)
            train = [all_seqs[i] for i in r.choice(len(all_seqs), n, replace=False)]
            train_dict = {i: v for i, v in enumerate(train)}
            train_ids = [np.array([sid[s] for s, _ in v]) for v in train]
            train_tuples = set(tuple(v.tolist()) for v in train_ids)
            SY.rng = np.random.default_rng(1000 + seed)
            model = SY.SemanticMarkov(alpha=0.1).fit(train_dict)
            synth = [np.array([sid[s] for s in SY.seq_symbols(model.sample(1000))])
                     for _ in range(N_SYNTH)]
            bt.append(tv(bigram(synth), bi_full))
            mu = [max(sim(R, s, rf) for rf in ref) for s in synth]
            dcr.append(1 - np.mean(mu))
            exact.append(sum(tuple(s.tolist()) in train_tuples for s in synth))
        rows.append({"n": n, "bigram_TV": round(np.mean(bt), 3),
                     "bigram_TV_sd": round(np.std(bt), 3),
                     "DCR_synth": round(float(np.mean(dcr)), 3),
                     "DCR_real_base": round(float(dcr_base), 3),
                     "exact_copies_of_train": round(float(np.mean(exact)), 1)})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "val_scaling.csv"), index=False)
    print("\n" + df.to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.errorbar(df.n, df.bigram_TV, yerr=df.bigram_TV_sd, fmt="o-", color="#2c7fb8", lw=2,
                 label="bigram TV (↓ better)")
    ax1.set_xlabel("training trajectories  n"); ax1.set_ylabel("bigram TV vs full real", color="#2c7fb8")
    ax1.set_xscale("log"); ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(df.n, df.DCR_synth, "s--", color="#d95f0e", lw=2, label="DCR synth→real")
    ax2.axhline(dcr_base, color="#555", ls=":", label=f"real→real DCR {dcr_base:.2f}")
    ax2.set_ylabel("DCR (↑ safer)", color="#d95f0e"); ax2.tick_params(axis="y", labelcolor="#d95f0e")
    ax1.set_title(f"Paris-581 · data-scaling at fixed vocabulary (m={M_TOP}, k={K})")
    fig.legend(loc="upper right", bbox_to_anchor=(.88, .88), fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG, "val_scaling.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("figure:", p)


if __name__ == "__main__":
    main()
