"""
Order-sensitive disclosure measure on Paris-581 (MAT-Sum m=2,k=5, order-1 Markov, the paper's
operating point), complementing the set-based, order-insensitive MUITAS similarity used
throughout the paper. MUITAS can score two sequences as maximally similar even when their semantic
states occur in a different order (it may flag representation-level collisions that are not
sequential memorisation), and, conversely, could in principle miss leakage of an ordered
subsequence, timing, or a distinctive routine that a strictly set-based measure cannot see.

We use normalized Levenshtein (edit) distance over the semantic-state sequence -- fully
order-sensitive -- as the complementary measure:
  sim_edit(s, x) = 1 - editdistance(s, x) / max(len(s), len(x))
and recompute, on the SAME N=200 synthetic sequences and the SAME real population used for the
paper's headline MUITAS-based numbers (Table "abstraction"/paris_disclosure.py), the same
exact-copy / near-copy / DCR triad, this time under sim_edit. We report, per synthetic sequence,
BOTH q_MUITAS(s) and q_edit(s), so we can directly quantify:
  (i)  MUITAS near-copies that are NOT edit near-copies (order-insensitive collisions), and
  (ii) edit near-copies that are NOT MUITAS near-copies (ordered leakage MUITAS would miss).
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS, matsum_synth as SY
from val_rq3 import rle, subset_matrix, sim as muitas_sim, markov

OUT = os.environ["MATSUM_OUT"]
M, K, N_SYN, R = 2, 5, 200, 20
REF_CAP = 300           # capped reference set for tractability (same convention as elsewhere)
TAU = 0.95
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t
    TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093


def edit_distance(a, b):
    """Levenshtein distance between two sequences of hashable tokens (numpy-vectorised rows)."""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    a = np.asarray(a); b = np.asarray(b)
    prev = np.arange(lb + 1)
    for i in range(1, la + 1):
        cur = np.empty(lb + 1, dtype=int)
        cur[0] = i
        cost = (b != a[i - 1]).astype(int)
        # cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1]+cost[j-1])  computed left-to-right
        for j in range(1, lb + 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost[j - 1])
        prev = cur
    return int(prev[lb])


def sim_edit(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = edit_distance(a, b)
    return 1.0 - d / max(len(a), len(b))


def main():
    print("[load] Paris-581, MAT-Sum m=2,k=5 operating point", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: list(g["sig2"]) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))
    rng = np.random.default_rng(0)

    rows = []
    for r in range(R):
        raw = {t: rle(per[t]) for t in tids}
        fs = Counter(s for t in tids for s in raw[t])
        freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
        freq_set = set(freq)
        remap = {s: (s if s in freq_set else max(freq, key=lambda f: jac(s, f))) for s in fs}
        real = [rle([remap[s] for s in raw[t]]) for t in tids]
        symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
        sid = {s: i for i, s in enumerate(symbols)}
        Rm = subset_matrix(symbols)
        real_ids = [np.array([sid[s] for s in v]) for v in real]

        grng = np.random.default_rng(1000 + r)
        synth = markov(real, N_SYN, grng)
        synth_ids = [np.array([sid[s] for s in v]) for v in synth]
        real_tuples = set(tuple(v) for v in real)

        ridx = grng.choice(len(real), min(REF_CAP, len(real)), replace=False)
        ref_ids = [real_ids[i] for i in ridx]
        ref_tok = [real[i] for i in ridx]     # token sequences (for edit distance, same alphabet as synth)

        for s_ids, s_tok in zip(synth_ids, synth):
            q_m = max((muitas_sim(Rm, s_ids, z) for z in ref_ids), default=0.0)
            q_e = max((sim_edit(s_tok, z) for z in ref_tok), default=0.0)
            exact = tuple(s_tok) in real_tuples
            rows.append({"rep": r, "q_muitas": q_m, "q_edit": q_e, "exact": exact})
        print(f"  replicate {r + 1}/{R} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "paris_edit_disclosure_raw.csv"), index=False)

    near_m = df.q_muitas >= TAU
    near_e = df.q_edit >= TAU
    only_m = (near_m & ~near_e)
    only_e = (near_e & ~near_m)
    both = (near_m & near_e)

    def pct_ci(mask):
        per_rep = df.assign(flag=mask).groupby("rep")["flag"].mean() * 100
        return per_rep.mean(), TCRIT * per_rep.std(ddof=1) / np.sqrt(R)

    m_mean, m_ci = pct_ci(near_m)
    e_mean, e_ci = pct_ci(near_e)
    both_mean, both_ci = pct_ci(both)
    onlym_mean, onlym_ci = pct_ci(only_m)
    onlye_mean, onlye_ci = pct_ci(only_e)
    exact_mean = df.groupby("rep")["exact"].sum().mean()
    exact_ci = TCRIT * df.groupby("rep")["exact"].sum().std(ddof=1) / np.sqrt(R)

    dcr_m = 1 - df.q_muitas; dcr_e = 1 - df.q_edit
    summary = pd.DataFrame([{
        "near_copy_muitas_pct": round(m_mean, 3), "near_copy_muitas_ci": round(m_ci, 3),
        "near_copy_edit_pct": round(e_mean, 3), "near_copy_edit_ci": round(e_ci, 3),
        "both_pct": round(both_mean, 3), "both_ci": round(both_ci, 3),
        "muitas_only_pct": round(onlym_mean, 3), "muitas_only_ci": round(onlym_ci, 3),
        "edit_only_pct": round(onlye_mean, 3), "edit_only_ci": round(onlye_ci, 3),
        "exact_copies_mean": round(exact_mean, 3), "exact_copies_ci": round(exact_ci, 3),
        "DCR_muitas_mean": round(float(dcr_m.mean()), 4), "DCR_edit_mean": round(float(dcr_e.mean()), 4),
        "DCR_muitas_p05": round(float(np.percentile(dcr_m, 5)), 4),
        "DCR_edit_p05": round(float(np.percentile(dcr_e, 5)), 4),
    }])
    summary.to_csv(os.path.join(OUT, "paris_edit_disclosure_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== Paris order-sensitive disclosure (MUITAS vs normalized edit distance), N=200, R=20, tau=0.95 ===")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
