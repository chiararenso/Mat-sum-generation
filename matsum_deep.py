"""
MAT-SUM -> synthetic MAT dataset.  PHASE 1b - deep generator + memorization test.

A small LSTM language model over the semantic-symbol alphabet, compared head-to-head
with the order-1 Markov baseline (Phase 1a) at the SAME (m, k) and the SAME empirical
dwell-time model, so the only thing that changes is the sequence generator.

The research question is a PRIVACY one: does the deep model buy fidelity by memorizing?
We report both fidelity (bigram TV, MUITAS->real) and leakage (exact copies, DCR, novel
bigrams) for the two models.
Run after matsum_prepare.py + matsum_summarize.py.
"""
import os
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matsum_summarize as MS
import matsum_synth as SY

torch.manual_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))
FIG = os.environ.get("MATSUM_FIG", os.path.join(HERE, "figures"))
M_TOP, K_SUPPORT, N_SYNTH = 3, 5, 200
EMB, HID, EPOCHS, LR = 32, 64, 500, 5e-3


class LSTMLM(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x, h=None):
        e = self.emb(x)
        y, h = self.lstm(e, h)
        return self.out(y), h


def main():
    print(f"[setup] vocabulary (m={M_TOP}, k={K_SUPPORT})")
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas)
    raw = SY.build_visits(sem, M_TOP)
    seqs, support, _, _ = SY.anonymize_vocabulary(raw, K_SUPPORT)
    symbols = sorted({s for v in seqs.values() for s, _ in v}, key=lambda z: sorted(z))
    S = len(symbols); START, END, PAD = S, S + 1, S + 2
    sid = {s: i for i, s in enumerate(symbols)}
    dwell = defaultdict(list)
    for v in seqs.values():
        for s, d in v:
            dwell[s].append(d)
    print(f"  alphabet={S}, real sequences={len(seqs)}")

    # encode: input [START,a,b,..], target [a,b,..,END]; pad to maxlen
    enc = [[sid[s] for s, _ in v] for v in seqs.values()]
    maxlen = max(len(e) for e in enc) + 1
    X = np.full((len(enc), maxlen), PAD); Y = np.full((len(enc), maxlen), PAD)
    for i, e in enumerate(enc):
        X[i, 0] = START; X[i, 1:1+len(e)] = e
        Y[i, :len(e)] = e; Y[i, len(e)] = END
    X = torch.tensor(X); Y = torch.tensor(Y)

    print(f"[train] LSTM ({EPOCHS} epochs)")
    model = LSTMLM(S + 3, EMB, HID)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    for ep in range(EPOCHS):
        opt.zero_grad()
        logits, _ = model(X)
        loss = lossf(logits.reshape(-1, S + 3), Y.reshape(-1))
        loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:4d}  loss {loss.item():.3f}")

    print("[sample] LSTM synthetic sequences")
    model.eval()
    lens = [len(e) for e in enc]
    lstm_seqs = {}
    with torch.no_grad():
        for i in range(N_SYNTH):
            x = torch.tensor([[START]]); h = None; out = []
            for _ in range(maxlen):
                logits, h = model(x, h)
                p = torch.softmax(logits[0, -1], dim=-1).numpy()
                nxt = np.random.choice(len(p), p=p)
                if nxt in (END, PAD, START) or nxt >= S:
                    break
                out.append(symbols[nxt]); x = torch.tensor([[nxt]])
            if not out:  # fallback
                out = [symbols[np.random.randint(S)]]
            lstm_seqs[f"lstm_{i:04d}"] = [[s, float(np.random.choice(dwell[s]))] for s in out]

    # Markov baseline at same (m,k)
    SY.rng = np.random.default_rng(42)
    mk = SY.SemanticMarkov(alpha=0.1).fit(seqs)
    mk_seqs = {f"mk_{i:04d}": mk.sample(120) for i in range(N_SYNTH)}

    # ---- evaluate both ----
    real_list = list(seqs.values())
    db_r = SY.dist_bigram(seqs); du_r = SY.dist_unigram(seqs)

    def evaluate(synth, name):
        du_s, db_s = SY.dist_unigram(synth), SY.dist_bigram(synth)
        mu = [max((SY.muitas_sim(s, r) for r in real_list), default=0.0) for s in synth.values()]
        exact = sum(any(SY.seq_symbols(s) == SY.seq_symbols(r) for r in real_list)
                    for s in synth.values())
        # near-copy: MUITAS >= 0.95 to some real
        near = sum(m >= 0.95 for m in mu)
        novel = 1 - len(set(db_s) & set(db_r)) / (len(set(db_s)) or 1)
        return {"model": name, "alphabet": len(symbols),
                "unigram_TV": round(SY.tv_distance(du_r, du_s), 3),
                "bigram_TV": round(SY.tv_distance(db_r, db_s), 3),
                "MUITAS_to_real": round(float(np.mean(mu)), 3),
                "DCR_mean": round(float(1 - np.mean(mu)), 3),
                "exact_copies": exact, "near_copies_.95": near,
                "novel_bigrams_%": round(novel * 100, 1),
                "len_mean": round(np.mean([len(v) for v in synth.values()]), 1)}

    # real->real baseline DCR
    dcr_real = []
    for i, r in enumerate(real_list):
        others = real_list[:i] + real_list[i+1:]
        dcr_real.append(1 - max((SY.muitas_sim(r, o) for o in others), default=0.0))
    rows = [evaluate(mk_seqs, "Markov (1a)"), evaluate(lstm_seqs, "LSTM (1b)")]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "deep_vs_markov.csv"), index=False)
    print(f"\n  real->real DCR baseline mean = {np.mean(dcr_real):.3f}\n")
    print(df.to_string(index=False))

    _figure(rows, np.mean(dcr_real))
    print("\n[done] output/deep_vs_markov.csv ; figures/deep_vs_markov.png")


def _figure(rows, dcr_base):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    names = [r["model"] for r in rows]; col = ["#2c7fb8", "#d95f0e"]
    ax[0].bar(names, [r["bigram_TV"] for r in rows], color=col)
    ax[0].set_title("Fidelity: bigram TV  (lower better)")
    ax[1].bar(names, [r["DCR_mean"] for r in rows], color=col)
    ax[1].axhline(dcr_base, color="k", ls="--", label=f"real→real {dcr_base:.2f}")
    ax[1].set_title("Privacy: DCR  (higher = safer)"); ax[1].legend()
    ax[2].bar(names, [r["exact_copies"] for r in rows], color=col, label="exact")
    ax[2].bar(names, [r["near_copies_.95"] for r in rows], color=col, alpha=.4,
              bottom=[r["exact_copies"] for r in rows], label="near (MUITAS≥.95)")
    ax[2].set_title("Leakage: copied trajectories"); ax[2].legend()
    fig.suptitle("Phase 1b — deep (LSTM) vs mechanistic (Markov): fidelity vs privacy", fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIG, "deep_vs_markov.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  figure:", p)


if __name__ == "__main__":
    main()
