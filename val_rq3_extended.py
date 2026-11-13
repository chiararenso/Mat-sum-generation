"""
RQ3 robustness: (1) n-gram fidelity up to order 5 for the RQ3 generators, and (2) an LSTM capacity scale.

Same protocol as val_rq3.py / val_rq3_order2.py: MAT-Sum m=2,k=5, nested paired samples (permutation seed =
replicate), N=200 synthetic sequences, 20 replicates, n in {20,50,100,300,581}. The synthetic sequences of
the original RQ3 runs were not saved, so they are REGENERATED here with the same seeds (the Markov and
base-LSTM rows can be checked against val_rq3_raw.csv / val_rq3_order2_raw.csv: --check).

Models (--models, comma separated):
  markov1, markov2l (light back-off), markov2h (heavy back-off),
  lstm_h64 (the original LSTM: 1 layer, hidden 64),
  lstm_h128, lstm_h256 (1 layer), lstm_l2 (2 layers, hidden 64)   [embedding 32, Adam 5e-3, 120 epochs for all]
Metrics per (rep, n, model), in the common OSM-label space against the real ground-truth label sequences:
  TV of n-grams for n=1..5; with --muitas 1 also near-copy %, dDCR (matched real-to-real baseline) and exact copies.
A real-vs-real floor (two disjoint halves of the paired real sample) is computed for the n-gram TVs.
"""
import os, sys, argparse, time
import numpy as np, pandas as pd
from collections import Counter, defaultdict
import torch, torch.nn as nn
import matsum_summarize as MS
from val_rq3 import (rle, ngram, tv, subset_matrix, sim, markov, jac, K, N_SYN, MAXLEN, NS, R, REF_CAP, NEAR, EPOCHS)
from val_rq3_order2 import markov2

OUT = os.environ["MATSUM_OUT"]
ORDERS = [1, 2, 3, 4, 5]
LSTM_CFG = {"lstm_h64": (64, 1), "lstm_h128": (128, 1), "lstm_h256": (256, 1), "lstm_l2": (64, 2)}


class LSTMLM(nn.Module):
    def __init__(self, vocab, emb=32, hid=64, layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x, h=None):
        e = self.emb(x); y, h = self.lstm(e, h); return self.out(y), h


def markov1(real, n, grng):
    """Order-1 Markov exactly as val_rq3.markov, but with a fixed (sorted) state order, so that results do not
    depend on the per-process hash seed (the original takes the states from a set)."""
    START = "<S>"; trans = defaultdict(Counter); states = set()
    for v in real:
        prev = START
        for s in v:
            trans[prev][s] += 1; states.add(s); prev = s
    states = sorted(states, key=lambda z: sorted(z)); lengths = [len(v) for v in real if len(v)] or [1]

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


def lstm_gen(real, symbols, sid, lengths, n, seed, hid, layers):
    torch.manual_seed(seed); S = len(symbols); START, PAD, V = S, S + 1, S + 2
    enc = [[sid[s] for s in v][:MAXLEN] for v in real]
    ml = max(len(e) for e in enc) + 1
    X = np.full((len(enc), ml), PAD); Y = np.full((len(enc), ml), PAD)
    for i, e in enumerate(enc):
        X[i, 0] = START; X[i, 1:1 + len(e)] = e; Y[i, :len(e)] = e
    X = torch.tensor(X); Y = torch.tensor(Y)
    model = LSTMLM(V, hid=hid, layers=layers); opt = torch.optim.Adam(model.parameters(), lr=5e-3)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="markov1,markov2l,markov2h,lstm_h64")
    ap.add_argument("--tag", default="ngrams")
    ap.add_argument("--muitas", type=int, default=0)
    ap.add_argument("--reps", type=int, default=R)
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--resume", type=int, default=0, help="continue from the replicates already saved in the CSV")
    ap.add_argument("--start", type=int, default=0, help="first replicate to run (to split work between processes)")
    ap.add_argument("--end", type=int, default=None, help="last replicate + 1 (default: --reps)")
    a = ap.parse_args()
    models = a.models.split(","); ns = [int(x) for x in a.ns.split(",")]
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

    print("[load] Paris-581", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: (list(g["sig2"]), list(g["top1"])) for t, g in sem.groupby("tid", sort=False)}
    tids = np.array(list(per.keys()))

    rows, t_start, first_rep = [], time.time(), 0
    csv_path = os.path.join(OUT, f"val_rq3_ext_{a.tag}_raw.csv")
    if a.resume and os.path.exists(csv_path):
        prev = pd.read_csv(csv_path); done = prev.groupby("rep").size()
        full = sorted(r_ for r_, c_ in done.items() if c_ == len(ns) * (len(models) + 1))   # complete replicates only
        prev = prev[prev.rep.isin(full)]; rows = prev.to_dict("records"); first_rep = (max(full) + 1) if full else 0
        print(f"[resume] {len(full)} complete replicates found, restarting at replicate {first_rep + 1}", flush=True)
    for r in range(max(first_rep, a.start), a.end if a.end is not None else a.reps):
        perm = np.random.default_rng(r).permutation(tids)          # SAME samples as RQ2/RQ3
        for n in ns:
            sample = list(perm[:n])
            grng = np.random.default_rng(10_000 * r + n)
            seed = 1000 * r + n
            # ---- representation: identical to val_rq3.run_config ----
            raw = {t: rle(per[t][0]) for t in sample}
            fs = Counter(s for t in sample for s in raw[t])
            frequent = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
            remap = {s: (s if s in frequent else max(frequent, key=lambda f: jac(s, f))) for s in fs}
            real = [rle([remap[s] for s in raw[t]])[:MAXLEN] for t in sample]
            dom = defaultdict(Counter); gt = []
            for t in sample:
                sig_l, top_l = per[t]
                for s, l in zip(sig_l, top_l):
                    dom[remap[s]][l] += 1
                gt.append(rle(top_l))
            dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}
            symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
            sid = {s: i for i, s in enumerate(symbols)}
            lengths = [len(v) for v in real if len(v)]
            gt_ng = {k: ngram(gt, k) for k in ORDERS}
            ref_idx = grng.choice(len(real), min(REF_CAP, len(real)), replace=False)   # same draw as the original (consumes grng)
            if a.muitas:
                Rm = subset_matrix(symbols)
                real_ids = [np.array([sid[s] for s in v]) for v in real]
                real_tup = set(tuple(v) for v in real)
                ref = [real_ids[i] for i in ref_idx]
                dcr_base = float(np.mean([1 - max(sim(Rm, ref[i], ref[j]) for j in range(len(ref)) if j != i)
                                          for i in range(len(ref))]))

            def evaluate(synth):
                synth_lab = [rle([dom[s] for s in v]) for v in synth]
                out = {f"TV{k}": tv(ngram(synth_lab, k), gt_ng[k]) for k in ORDERS}
                if a.muitas:
                    synth_ids = [np.array([sid[s] for s in v]) for v in synth]
                    max_all = [max((sim(Rm, s, r_) for r_ in real_ids), default=0.0) for s in synth_ids]
                    max_ref = [max((sim(Rm, s, r_) for r_ in ref), default=0.0) for s in synth_ids]
                    dcr_s = float(1 - np.mean(max_ref))
                    out.update({"near_pct": 100.0 * float(np.mean([m >= NEAR for m in max_all])),
                                "dDCR": dcr_s - dcr_base, "DCR_base": dcr_base,
                                "exact_copies": sum(tuple(v) in real_tup for v in synth)})
                return out

            # real-vs-real floor for the n-gram TVs
            frng = np.random.default_rng(7000 + 31 * r + n)
            pp = frng.permutation(len(gt)); h = len(gt) // 2
            A_, B_ = [gt[i] for i in pp[:h]], [gt[i] for i in pp[h:2 * h]]
            rows.append({"rep": r, "n": n, "model": "real-vs-real",
                         **{f"TV{k}": tv(ngram(A_, k), ngram(B_, k)) for k in ORDERS}})

            for m in models:
                if m == "markov1":
                    synth = markov1(real, N_SYN, grng)
                elif m == "markov2l":
                    synth = markov2(real, N_SYN, np.random.default_rng(20_000 * r + n + 0), None)
                elif m == "markov2h":
                    synth = markov2(real, N_SYN, np.random.default_rng(20_000 * r + n + 7), "heavy")
                else:
                    hid, layers = LSTM_CFG[m]
                    synth = lstm_gen(real, symbols, sid, lengths, N_SYN, seed, hid, layers)
                rows.append({"rep": r, "n": n, "model": m, **evaluate(synth)})
            print(f"  replicate {r + 1}/{a.reps} n={n} done  ({(time.time() - t_start) / 60:.1f} min)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"val_rq3_ext_{a.tag}_raw.csv"), index=False)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(os.path.join(OUT, f"val_rq3_ext_{a.tag}_raw.csv"), index=False)
    if a.check:
        old = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "paris", "val_rq3_raw.csv"))
        old2 = pd.read_csv(os.path.join(OUT, "val_rq3_order2_raw.csv"))
        for m, src, name in [("markov1", old, "Markov"), ("lstm_h64", old, "LSTM"), ("markov2l", old2, "Markov2"), ("markov2h", old2, "Markov2_heavy")]:
            if m not in models:
                continue
            for _, row in raw_df[raw_df.model == m].iterrows():
                o = src[(src.rep == row.rep) & (src.n == row.n) & (src.model == name)]
                if m == 'markov1':
                    continue
                print(f"CHECK {m:9s} rep={int(row.rep)} n={int(row.n)}: TV2 new={row.TV2:.6f} old={float(o.sem_genTV.iloc[0]):.6f}"
                      f" | TV3 new={row.TV3:.6f} old={float(o.sem_triTV.iloc[0]):.6f}")
    print("saved", os.path.join(OUT, f"val_rq3_ext_{a.tag}_raw.csv"))


if __name__ == "__main__":
    main()
