"""
EXP B' - deep vs mechanistic ACROSS n on Paris-581 (makes H3 decisive).
Fixed vocabulary (m=2,k=5). For each n, train BOTH an order-1 semantic Markov
and a small LSTM on the SAME n-trajectory subsample; measure fidelity (bigram TV
vs full real) and MEMORISATION (exact + near copies of the TRAINING set) with a
vectorised MUITAS. Expectation: the LSTM buys fidelity by memorising, worst at
small n; the Markov generalises (≈0 copies) at every n.
"""
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter
import matsum_summarize as MS, matsum_synth as SY
from matsum_deep import LSTMLM

torch.manual_seed(0); np.random.seed(0)
OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K = 2, 5
NS = [20, 50, 100, 300, 581]
SEEDS = [0, 1]
N_SYNTH, MAXLEN, EPOCHS, REF_CAP = 150, 100, 200, 150


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
    M = R[np.ix_(a, b)]
    return (M.max(1).sum() + M.max(0).sum()) / (len(a) + len(b))


def tv(pa, pb):
    return 0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))


def bigram(id_seqs):
    c = Counter()
    for v in id_seqs:
        for a, b in zip(v, v[1:]):
            c[(a, b)] += 1
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def train_lstm(train_ids, S):
    START, END, PAD = S, S + 1, S + 2
    enc = [list(e) for e in train_ids]
    ml = max(len(e) for e in enc) + 1
    X = np.full((len(enc), ml), PAD); Y = np.full((len(enc), ml), PAD)
    for i, e in enumerate(enc):
        X[i, 0] = START; X[i, 1:1+len(e)] = e
        Y[i, :len(e)] = e; Y[i, len(e)] = END
    X = torch.tensor(X); Y = torch.tensor(Y)
    model = LSTMLM(S + 3, 32, 64); opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lf = nn.CrossEntropyLoss(ignore_index=PAD)
    for _ in range(EPOCHS):
        opt.zero_grad(); logits, _ = model(X)
        loss = lf(logits.reshape(-1, S + 3), Y.reshape(-1)); loss.backward(); opt.step()
    return model, START, END, PAD


def sample_lstm(model, S, START, END, PAD, lengths):
    model.eval(); out = []
    with torch.no_grad():
        for _ in range(N_SYNTH):
            x = torch.tensor([[START]]); h = None; seq = []
            for _ in range(MAXLEN):
                logits, h = model(x, h)
                p = torch.softmax(logits[0, -1], -1).numpy()
                nxt = np.random.choice(len(p), p=p)
                if nxt >= S:
                    break
                seq.append(nxt); x = torch.tensor([[nxt]])
            out.append(np.array(seq if seq else [np.random.randint(S)]))
    return out


def main():
    gdf, areas = MS.load(); sem = MS.semantic_mapping(gdf, areas)
    raw = SY.build_visits(sem, M); seqs_full, support, _, _ = SY.anonymize_vocabulary(raw, K)
    symbols = sorted({s for v in seqs_full.values() for s, _ in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; S = len(symbols); R = subset_matrix(symbols)
    all_seqs = list(seqs_full.values())
    ids = lambda v: np.array([sid[s] for s, _ in v][:MAXLEN])
    real_full = [ids(v) for v in all_seqs]; bi_full = bigram(real_full)
    print(f"vocab={S} symbols, {len(all_seqs)} individuals", flush=True)

    rows = []
    for n in NS:
        acc = {("Markov", k): [] for k in ("bt", "exact", "near")}
        acc.update({("LSTM", k): [] for k in ("bt", "exact", "near")})
        for seed in SEEDS:
            r = np.random.default_rng(seed)
            idx = r.choice(len(all_seqs), n, replace=False)
            train_sym = [[(s, d) for s, d in all_seqs[i]][:MAXLEN] for i in idx]
            train_ids = [np.array([sid[s] for s, _ in v]) for v in train_sym]
            train_tuples = set(tuple(v.tolist()) for v in train_ids)

            def evaluate(synth, tag):
                acc[(tag, "bt")].append(tv(bigram(synth), bi_full))
                mx = [max((sim(R, s, t) for t in train_ids), default=0.0) for s in synth]
                acc[(tag, "exact")].append(sum(tuple(s.tolist()) in train_tuples for s in synth))
                acc[(tag, "near")].append(100 * np.mean([m >= 0.95 for m in mx]))

            # Markov
            SY.rng = np.random.default_rng(1000 + seed)
            mk = SY.SemanticMarkov(alpha=0.1).fit({i: v for i, v in enumerate(train_sym)})
            mk_s = [np.array([sid[s] for s in SY.seq_symbols(mk.sample(MAXLEN))]) for _ in range(N_SYNTH)]
            evaluate(mk_s, "Markov")
            # LSTM
            torch.manual_seed(seed); np.random.seed(seed)
            model, START, END, PAD = train_lstm(train_ids, S)
            ls_s = sample_lstm(model, S, START, END, PAD, [len(e) for e in train_ids])
            evaluate(ls_s, "LSTM")

        for tag in ("Markov", "LSTM"):
            rows.append({"n": n, "model": tag,
                         "bigram_TV": round(float(np.mean(acc[(tag, "bt")])), 3),
                         "exact_copies": round(float(np.mean(acc[(tag, "exact")])), 1),
                         "near_copies_pct": round(float(np.mean(acc[(tag, "near")])), 1)})
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "val_deep_scaling.csv"), index=False)
    print("\n" + df.to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for tag, col in [("Markov", "#2c7fb8"), ("LSTM", "#d95f0e")]:
        d = df[df.model == tag]
        ax[0].plot(d.n, d.bigram_TV, "o-", color=col, lw=2, label=tag)
        ax[1].plot(d.n, d.near_copies_pct, "s-", color=col, lw=2, label=tag)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("training individuals  n"); a.legend()
    ax[0].set_title("Fidelity: bigram TV (↓ better)")
    ax[1].set_title("Memorisation: near-copies of training (%)")
    fig.suptitle("Paris · deep vs mechanistic across n  (m=2, k=5)", fontsize=13)
    fig.tight_layout(); p = os.path.join(FIG, "val_deep_scaling.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("figure:", p)


if __name__ == "__main__":
    main()
