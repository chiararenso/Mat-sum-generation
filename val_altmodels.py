"""
Alternative sequence generators on Paris-581 (m=2,k=5), same metrics as the sweep.
  - VOMM : variable-order Markov via Jelinek-Mercer INTERPOLATION of orders 1..K
    (cannot do worse than order-1 on bigrams; the gain shows on TRIGRAM statistics).
  - DP-Markov : order-1 with Laplace noise on transition counts + per-user contribution
    clamping (bounds L1 sensitivity) -> epsilon-DP; swept over a wide epsilon range.
Baseline: order-1 Markov. Fidelity = bigram-TV AND trigram-TV vs full real; privacy = DCR vs
real-real baseline + exact/near copies of training. Vectorised MUITAS (subset matrix).

Why trigram-TV: an order-1 Markov is parameterised exactly by bigram counts, so it already
matches the bigram distribution near-optimally; a higher-order model can only demonstrate its
advantage on longer-range statistics.

DP note: clamping each user to CAP_TOTAL transitions bounds the histogram L1 sensitivity to
CAP_TOTAL; Laplace(CAP_TOTAL/eps) then gives user-level eps-DP. Prototype mechanism (cf.
AdaTrace/PrivTrace for tuned, adaptive treatments), not a tuned system.
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS, matsum_synth as SY

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K_ANON = 2, 5
KMAX = 3
ALPHA = 0.1
LAMBDAS = {2: 0.45, 1: 0.50}   # interpolation weights for context lengths (order-3, order-2); rest -> uniform
N_SYNTH, REF_CAP = 200, 150
CAP_TOTAL = 200
EPS_LIST = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, np.inf]
rng = np.random.default_rng(0)


def subset_matrix(symbols):
    S = len(symbols); R = np.zeros((S, S), bool)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            if b <= a:
                R[i, j] = True
    return R


def sim(R, a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    Msub = R[np.ix_(a, b)]
    return (Msub.max(1).sum() + Msub.max(0).sum()) / (len(a) + len(b))


def tv(pa, pb):
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))


def ngram(seqs, k):
    c = Counter()
    for v in seqs:
        for i in range(len(v) - k + 1):
            c[tuple(v[i:i+k])] += 1
    n = sum(c.values()) or 1
    return {t: x / n for t, x in c.items()}


def gen_markov1(train_ids, S, lengths, n):
    START = S; trans = defaultdict(Counter)
    for v in train_ids:
        prev = START
        for s in v:
            trans[prev][s] += 1; prev = s
    def nxt(state):
        c = np.array([trans[state][s] for s in range(S)], float)
        return rng.choice(S, p=(c + ALPHA) / (c.sum() + ALPHA * S))
    out = []
    for _ in range(n):
        L = int(rng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            s = nxt(st); seq.append(s); st = s
        out.append(np.array(seq))
    return out


def gen_vomm(train_ids, S, lengths, n):
    """Jelinek-Mercer interpolation over context lengths 1..KMAX-1 + uniform."""
    START = S
    counts = {L: defaultdict(Counter) for L in range(1, KMAX)}
    uni = Counter()
    for v in train_ids:
        seq = [START] + list(v)
        for i in range(1, len(seq)):
            uni[seq[i]] += 1
            for L in range(1, KMAX):
                if i - L >= 0:
                    counts[L][tuple(seq[i-L:i])][seq[i]] += 1
    uni_p = np.array([uni[s] for s in range(S)], float); uni_p = (uni_p + ALPHA); uni_p /= uni_p.sum()
    lam0 = 1.0 - sum(LAMBDAS.values())
    def dist(hist):
        p = lam0 * uni_p.copy()
        for L, w in LAMBDAS.items():
            comp = uni_p
            if len(hist) >= L:
                ctx = tuple(hist[-L:])
                if ctx in counts[L]:
                    c = np.array([counts[L][ctx][s] for s in range(S)], float)
                    if c.sum() > 0:
                        comp = c / c.sum()
            p = p + w * comp
        return p / p.sum()
    out = []
    for _ in range(n):
        L = int(rng.choice(lengths)); hist = [START]; seq = []
        for _ in range(L):
            s = rng.choice(S, p=dist(hist)); seq.append(s); hist.append(s)
        out.append(np.array(seq))
    return out


def gen_dp_markov(train_ids, S, lengths, n, eps):
    START = S; C = np.zeros((S + 1, S), float)
    for v in train_ids:
        tr = [(START if i == 0 else v[i-1], v[i]) for i in range(len(v))]
        if len(tr) > CAP_TOTAL:
            sel = rng.choice(len(tr), CAP_TOTAL, replace=False); tr = [tr[i] for i in sel]
        for a, b in tr:
            C[a, b] += 1
    if np.isfinite(eps):
        C = C + rng.laplace(0.0, CAP_TOTAL / eps, C.shape)
    C = np.clip(C, 0, None) + ALPHA
    P = C / C.sum(1, keepdims=True)
    out = []
    for _ in range(n):
        L = int(rng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            s = rng.choice(S, p=P[st]); seq.append(s); st = s
        out.append(np.array(seq))
    return out


def evaluate(synth, name, bi_full, tri_full, ref, R, train_ids, train_tuples):
    mu_ref = [max(sim(R, s, r) for r in ref) for s in synth]
    mu_tr = [max((sim(R, s, t) for t in train_ids), default=0.0) for s in synth]
    return {"model": name,
            "bigram_TV": round(tv(ngram(synth, 2), bi_full), 3),
            "trigram_TV": round(tv(ngram(synth, 3), tri_full), 3),
            "DCR_synth": round(float(1 - np.mean(mu_ref)), 3),
            "exact_copies": sum(tuple(s.tolist()) in train_tuples for s in synth),
            "near_copies_pct": round(float(100 * np.mean([m >= 0.95 for m in mu_tr])), 1)}


def main():
    gdf, areas = MS.load(); sem = MS.semantic_mapping(gdf, areas)
    raw = SY.build_visits(sem, M); seqs, support, _, _ = SY.anonymize_vocabulary(raw, K_ANON)
    symbols = sorted({s for v in seqs.values() for s, _ in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; S = len(symbols); R = subset_matrix(symbols)
    train_ids = [np.array([sid[s] for s, _ in v]) for v in seqs.values()]
    lengths = [len(v) for v in train_ids]
    bi_full, tri_full = ngram(train_ids, 2), ngram(train_ids, 3)
    train_tuples = set(tuple(v.tolist()) for v in train_ids)
    ref = [train_ids[i] for i in rng.choice(len(train_ids), min(REF_CAP, len(train_ids)), replace=False)]
    dcr_base = np.mean([1 - max(sim(R, ref[i], ref[j]) for j in range(len(ref)) if j != i)
                        for i in range(len(ref))])
    print(f"vocab={S}, individuals={len(train_ids)}, real-real DCR base={dcr_base:.3f}", flush=True)
    ev = lambda synth, name: evaluate(synth, name, bi_full, tri_full, ref, R, train_ids, train_tuples)

    rows = [ev(gen_markov1(train_ids, S, lengths, N_SYNTH), "Markov(1)")]; print(rows[-1], flush=True)
    rows.append(ev(gen_vomm(train_ids, S, lengths, N_SYNTH), f"VOMM-interp(<= {KMAX})")); print(rows[-1], flush=True)
    for eps in EPS_LIST:
        tag = "DP-Markov eps=inf" if not np.isfinite(eps) else f"DP-Markov eps={eps:g}"
        rows.append(ev(gen_dp_markov(train_ids, S, lengths, N_SYNTH, eps), tag)); print(rows[-1], flush=True)

    df = pd.DataFrame(rows); df["DCR_real_base"] = round(float(dcr_base), 3)
    df.to_csv(os.path.join(OUT, "val_altmodels.csv"), index=False)
    print("\n" + df.to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dp = df[df.model.str.startswith("DP-Markov") & (df.model != "DP-Markov eps=inf")].copy()
    dp["eps"] = [float(m.split("=")[1]) for m in dp.model]; dp = dp.sort_values("eps")
    fig, ax1 = plt.subplots(figsize=(7.8, 5))
    ax1.plot(dp.eps, dp.bigram_TV, "o-", color="#2c7fb8", lw=2, label="DP-Markov bigram TV (↓)")
    mk = df[df.model == "Markov(1)"].iloc[0]; vo = df[df.model.str.startswith("VOMM")].iloc[0]
    ax1.axhline(mk.bigram_TV, color="#2c7fb8", ls=":", lw=1.3, label=f"Markov(1) bi {mk.bigram_TV}")
    ax1.set_xlabel("privacy budget  ε  (smaller = more private)"); ax1.set_ylabel("bigram TV", color="#2c7fb8")
    ax1.set_xscale("log"); ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(dp.eps, dp.DCR_synth, "s--", color="#d95f0e", lw=2, label="DP-Markov DCR (↑ safer)")
    ax2.axhline(dcr_base, color="#555", ls=":", label=f"real→real DCR {dcr_base:.2f}")
    ax2.set_ylabel("DCR", color="#d95f0e")
    ax1.set_title("Paris-581 · DP-Markov privacy/fidelity vs ε  (m=2,k=5)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8, framealpha=.92)
    fig.tight_layout(); p = os.path.join(FIG, "val_altmodels.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("figure:", p)


if __name__ == "__main__":
    main()
