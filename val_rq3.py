"""
RQ3 -- Generator capacity & privacy-fidelity trade-off (Paris-581).
Same MAT-Sum representation, same nested paired user samples as RQ2 (permutation seed = replicate),
same m=2,k=5, same empirical length distribution, 200 synthetic per run. Two generators only:
  - order-1 Markov
  - LSTM (small language model over the same per-sample symbol vocabulary)
Everything except the generator is identical. Three main metrics per (n, replicate, model):
  1. semantic generation TV   (quality in the fixed OSM-label space)
  2. near-copy rate           (% of the 200 synth whose max MUITAS to ANY training user >= 0.95)
  3. dDCR = DCR_synth - DCR_base   (matched per-sample real-to-real baseline)
Exact copies / 200 are also recorded (secondary). Figure: 3 panels (A sem-gen-TV, B near-copy%,
C dDCR with zero line), Markov vs LSTM, mean +/- 95% CI over 20 paired replicates.
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS
import torch, torch.nn as nn

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K, N_SYN, MAXLEN = 2, 5, 200, 150
NS = [20, 50, 100, 300, 581]
R, REF_CAP, EPOCHS = 20, 150, 120
NEAR = 0.95
try:
    from scipy.stats import t as _t
    TCRIT = float(_t.ppf(0.975, df=R - 1))
except Exception:
    TCRIT = 2.093
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def rle(x):
    out = []
    for v in x:
        if not out or out[-1] != v:
            out.append(v)
    return out


def ngram(seqs, k):
    c = Counter()
    for v in seqs:
        for i in range(len(v) - k + 1):
            c[tuple(v[i:i+k])] += 1
    n = sum(c.values()) or 1
    return {t: x / n for t, x in c.items()}


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def subset_matrix(symbols):
    S = len(symbols); Rm = np.zeros((S, S), bool)
    for i, a in enumerate(symbols):
        for j, b in enumerate(symbols):
            if b <= a:
                Rm[i, j] = True
    return Rm


def sim(Rm, a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    m = Rm[np.ix_(a, b)]
    return (m.max(1).sum() + m.max(0).sum()) / (len(a) + len(b))


def markov(real, n, grng):
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = list(states); lengths = [len(v) for v in real if len(v)] or [1]
    def nxt(st):
        c = np.array([trans[st][s] for s in states], float)
        return states[grng.choice(len(states), p=(c + 0.1) / (c.sum() + 0.1 * len(states)))]
    out = []
    for _ in range(n):
        L = int(grng.choice(lengths)); st = START; seq = []
        for _ in range(L):
            st = nxt(st); seq.append(st)
        out.append(seq)
    return out


class LSTMLM(nn.Module):
    def __init__(self, vocab, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb); self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)
    def forward(self, x, h=None):
        e = self.emb(x); y, h = self.lstm(e, h); return self.out(y), h


def lstm_gen(real, symbols, sid, lengths, n, seed):
    torch.manual_seed(seed); S = len(symbols); START, PAD, V = S, S + 1, S + 2
    enc = [[sid[s] for s in v][:MAXLEN] for v in real]
    ml = max(len(e) for e in enc) + 1
    X = np.full((len(enc), ml), PAD); Y = np.full((len(enc), ml), PAD)
    for i, e in enumerate(enc):
        X[i, 0] = START; X[i, 1:1+len(e)] = e; Y[i, :len(e)] = e
    X = torch.tensor(X); Y = torch.tensor(Y)
    model = LSTMLM(V); opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lf = nn.CrossEntropyLoss(ignore_index=PAD)
    for _ in range(EPOCHS):
        opt.zero_grad(); logits, _ = model(X); lf(logits.reshape(-1, V), Y.reshape(-1)).backward(); opt.step()
    model.eval(); g = np.random.default_rng(seed + 1); out = []
    with torch.no_grad():
        for _ in range(n):
            L = int(g.choice(lengths)); x = torch.tensor([[START]]); h = None; seq = []
            for _ in range(L):
                logits, h = model(x, h); p = torch.softmax(logits[0, -1], -1).numpy()
                p[START] = 0; p[PAD] = 0; s = p.sum()
                nxt = g.choice(V, p=p / s) if s > 0 else g.integers(S)
                seq.append(symbols[nxt]); x = torch.tensor([[int(nxt)]])
            out.append(seq)
    return out


def run_config(tids, per, grng, seed):
    raw = {t: rle(per[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    frequent = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]])[:MAXLEN] for t in tids]
    dom = defaultdict(Counter); gt = []
    for t in tids:
        sig_l, top_l = per[t]
        for s, l in zip(sig_l, top_l):
            dom[remap[s]][l] += 1
        gt.append(rle(top_l))
    dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = subset_matrix(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]
    real_tup = set(tuple(v) for v in real)
    real_bi, gt_bi = ngram(real, 2), ngram(gt, 2)
    gt_tri = ngram(gt, 3)
    lengths = [len(v) for v in real if len(v)]
    ref = [real_ids[i] for i in grng.choice(len(real_ids), min(REF_CAP, len(real_ids)), replace=False)]
    dcr_base = float(np.mean([1 - max(sim(Rm, ref[i], ref[j]) for j in range(len(ref)) if j != i)
                              for i in range(len(ref))]))

    def evaluate(synth):
        synth_ids = [np.array([sid[s] for s in v]) for v in synth]
        max_all = [max((sim(Rm, s, r) for r in real_ids), default=0.0) for s in synth_ids]
        max_ref = [max((sim(Rm, s, r) for r in ref), default=0.0) for s in synth_ids]
        near = 100.0 * np.mean([m >= NEAR for m in max_all])
        dcr_s = float(1 - np.mean(max_ref))
        synth_lab = [rle([dom[s] for s in v]) for v in synth]
        return {"sem_genTV": tv(ngram(synth_lab, 2), gt_bi), "sem_triTV": tv(ngram(synth_lab, 3), gt_tri),
                "native_biTV": tv(ngram(synth, 2), real_bi),
                "near_pct": float(near), "DCR_synth": dcr_s, "DCR_base": dcr_base, "dDCR": dcr_s - dcr_base,
                "exact_copies": sum(tuple(v) in real_tup for v in synth)}

    out = {}
    out["Markov"] = evaluate(markov(real, N_SYN, grng))
    out["LSTM"] = evaluate(lstm_gen(real, symbols, sid, lengths, N_SYN, seed))
    return out


def main():
    print("[load] Paris-581 (fixed public map/aspects)")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: (list(g["sig2"]), list(g["top1"])) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))

    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(tids)     # SAME samples as RQ2
        for n in NS:
            grng = np.random.default_rng(10_000 * r + n)
            res = run_config(list(perm[:n]), per, grng, seed=1000 * r + n)
            for model in ("Markov", "LSTM"):
                rows.append({"rep": r, "n": n, "model": model, **res[model]})
            print(f"  replicate {r+1}/{R}  n={n} done", flush=True)
    raw_df = pd.DataFrame(rows); raw_df.to_csv(os.path.join(OUT, "val_rq3_raw.csv"), index=False)

    metrics = ["sem_genTV", "sem_triTV", "near_pct", "dDCR", "exact_copies", "native_biTV", "DCR_synth", "DCR_base"]
    agg = []
    for n in NS:
        for model in ("Markov", "LSTM"):
            sub = raw_df[(raw_df.n == n) & (raw_df.model == model)]; row = {"n": n, "model": model}
            for m in metrics:
                a = sub[m].to_numpy(float)
                row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
            agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "val_rq3_summary.csv"), index=False)
    pd.set_option("display.width", 240)
    print("\n=== summary (mean +/- 95% CI) ===")
    print(adf[["n", "model", "sem_genTV", "sem_triTV", "near_pct", "dDCR", "exact_copies"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.array(NS); fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    col = {"Markov": "#2c7fb8", "LSTM": "#d95f0e"}
    for model in ("Markov", "LSTM"):
        d = adf[adf.model == model]
        ax[0].errorbar(x, d.sem_genTV, yerr=d.sem_genTV_ci, fmt="o-", color=col[model], capsize=3, lw=2, label=model)
        ax[1].errorbar(x, d.near_pct, yerr=d.near_pct_ci, fmt="s-", color=col[model], capsize=3, lw=2, label=model)
        ax[2].errorbar(x, d.dDCR, yerr=d.dDCR_ci, fmt="D-", color=col[model], capsize=3, lw=2, label=model)
    ax[0].set_title("(A) Semantic generation TV"); ax[0].set_ylabel("TV (↓ better)")
    ax[1].set_title(f"(B) Near-copy rate (MUITAS≥{NEAR})"); ax[1].set_ylabel("% of 200 (↓ safer)")
    ax[2].set_title("(C) ΔDCR"); ax[2].set_ylabel("DCR$_{synth}$ − DCR$_{base}$ (↑ safer)"); ax[2].axhline(0, color="#888", lw=1, ls=":")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off(); a.legend(fontsize=9)
    fig.suptitle(f"RQ3 · generator capacity (Markov vs LSTM) · same MAT-Sum rep · {R} paired replicates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "val_rq3.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
