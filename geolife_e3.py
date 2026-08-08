"""
E3 -- generator capacity on GeoLife (external replication of RQ3): order-1 Markov vs a light LSTM,
on the SAME MAT-Sum representation and the SAME nested paired user samples as E2.
n in {20,50,100,152}, R=20 nested paired replicates (perm = default_rng(r).permutation(users), prefixes).
Per (r,n) we rebuild the MAT-Sum vocabulary on the sampled users, estimate BOTH generators on the sample,
and generate 200 synthetic SESSIONS each (no cross-session transitions; length drawn from empirical L).

Metrics per generator (mean +/- 95% CI over the 20 replicates):
  semantic generation TV (label-space bigram)  -- fidelity
  semantic generation TV (label-space trigram)  -- higher-order fidelity
  near-copy rate: % synth with max MUITAS to training >= 0.95  -- record-level disclosure
  exact session copies /200
  user-aware DCR (synth, real->real baseline, dDCR)
"""
import os, time, numpy as np, pandas as pd
from collections import Counter, defaultdict
from geolife_e1 import load, rle, bigram, tv, submat, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, N_SYN, R, REF, NEAR = 5, 200, 20, 400, 0.95
NS = [20, 50, 100, 152]
EPOCHS, HID, EMB, LSTM_BUDGET_S = 80, 64, 32, 25.0
N_TRAIN, MAXLEN, BATCH = 1500, 40, 128   # cap LSTM training set (user-balanced), seq length, minibatch
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093

import torch, torch.nn as nn


def trigram(seqs):
    c = Counter()
    for v in seqs:
        for a, b, d in zip(v, v[1:], v[2:]):
            c[(a, b, d)] += 1
    n = sum(c.values()) or 1
    return {k: x / n for k, x in c.items()}


class LSTMGen(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V + 2, EMB)          # +2: START, PAD
        self.lstm = nn.LSTM(EMB, HID, batch_first=True)
        self.fc = nn.Linear(HID, V + 2)

    def forward(self, x, h=None):
        e = self.emb(x); o, h = self.lstm(e, h); return self.fc(o), h


def lstm_gen(real, real_user, n, grng, tseed):
    """Train a small LSTM over state-id sequences; sample n sessions (length ~ empirical L).
    Training set is user-balanced (round-robin over users, cap N_TRAIN), length-capped (MAXLEN),
    and minibatched -- avoids the OOM from a single full-batch forward at large n."""
    st = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i + 2 for i, s in enumerate(st)}       # 0=PAD, 1=START
    V = len(st)
    L = [len(v) for v in real if len(v)] or [1]      # empirical lengths for generation (untruncated)
    pairs = [([1] + [sid[s] for s in v][:MAXLEN], u) for v, u in zip(real, real_user) if v]
    if V < 2 or len(pairs) < 2:
        return markov(real, n, grng)                 # degenerate -> fall back
    # user-balanced round-robin subsample up to N_TRAIN
    by_user = defaultdict(list)
    for i, (_, u) in enumerate(pairs):
        by_user[u].append(i)
    for u in by_user:
        grng.shuffle(by_user[u])
    order = []
    us = sorted(by_user); pos = {u: 0 for u in us}
    while len(order) < min(N_TRAIN, len(pairs)):
        adv = False
        for u in us:
            if pos[u] < len(by_user[u]):
                order.append(by_user[u][pos[u]]); pos[u] += 1; adv = True
                if len(order) >= min(N_TRAIN, len(pairs)):
                    break
        if not adv:
            break
    seqs = [pairs[i][0] for i in order]
    Lm = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), Lm), int)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
    Xt = torch.tensor(X)
    torch.manual_seed(tseed)
    net = LSTMGen(V); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    t0 = time.time(); nb = int(np.ceil(len(seqs) / BATCH))
    bg = np.random.default_rng(tseed)
    for ep in range(EPOCHS):
        perm = bg.permutation(len(seqs))
        for b in range(nb):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            xb = Xt[idx]
            opt.zero_grad()
            logit, _ = net(xb[:, :-1])
            loss = loss_fn(logit.reshape(-1, V + 2), xb[:, 1:].reshape(-1))
            loss.backward(); opt.step()
        if time.time() - t0 > LSTM_BUDGET_S:
            break
    inv = {v: k for k, v in sid.items()}
    net.eval(); out = []
    tg = torch.Generator().manual_seed(tseed + 1)
    with torch.no_grad():
        for _ in range(n):
            ln = int(grng.choice(L)); h = None
            cur = torch.tensor([[1]]); seq = []
            for _ in range(ln):
                logit, h = net(cur, h)
                p = torch.softmax(logit[0, -1, :], 0); p[0] = 0; p[1] = 0
                p = p / p.sum()
                idx = int(torch.multinomial(p, 1, generator=tg))
                seq.append(inv.get(idx, st[0])); cur = torch.tensor([[idx]])
            out.append(rle(seq))
    return out


def evaluate(synth, sid, Rm, dom, gt_bi, gt_tri, real_tup, ref_ids, ref_user, real_ids, real_user, grng):
    synth_ids = [np.array([sid[s] for s in v if s in sid]) for v in synth]
    synth_lab = [rle([dom[s] for s in v if s in dom]) for v in synth]
    sem_bi = tv(bigram(synth_lab), gt_bi)
    sem_tri = tv(trigram(synth_lab), gt_tri)
    exact = sum(tuple(v) in real_tup for v in synth)
    near = 100.0 * np.mean([max((sim(Rm, s, r) for r in ref_ids), default=0.0) >= NEAR
                            for s in synth_ids]) if synth_ids else 0.0
    dcr_s = float(np.mean([1 - max((sim(Rm, s, r) for r in ref_ids), default=0.0) for s in synth_ids]))
    # real->real user-aware baseline
    qidx = grng.choice(len(real_ids), min(REF, len(real_ids)), replace=False)
    per_user = defaultdict(list)
    for qi in qidx:
        u = real_user[qi]
        best = max((sim(Rm, real_ids[qi], ref_ids[j]) for j in range(len(ref_ids)) if ref_user[j] != u), default=0.0)
        per_user[u].append(1 - best)
    dcr_b = float(np.mean([np.mean(v) for v in per_user.values()])) if per_user else 0.0
    return {"sem_biTV": sem_bi, "sem_triTV": sem_tri, "near_pct": float(near), "exact_copies": exact,
            "DCR_synth": dcr_s, "DCR_base": dcr_b, "dDCR": dcr_s - dcr_b}


def run(sess, users, grng, tseed):
    tids = [t for t in sess if sess[t][2] in users]
    raw = {t: rle(sess[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids]
    real_user = [sess[t][2] for t in tids]
    dom = defaultdict(Counter); gt = []
    for t in tids:
        sig_l, top_l, _ = sess[t]
        for s, l in zip(sig_l, top_l):
            dom[remap[s]][l] += 1
        gt.append(rle(top_l))
    dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = submat(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]
    gt_bi, gt_tri = bigram(gt), trigram(gt)
    real_tup = set(tuple(v) for v in real)
    ridx = grng.choice(len(real_ids), min(REF, len(real_ids)), replace=False)
    ref_ids = [real_ids[i] for i in ridx]; ref_user = [real_user[i] for i in ridx]

    gens = {"Markov": markov(real, N_SYN, grng),
            "LSTM": lstm_gen(real, real_user, N_SYN, grng, tseed)}
    res = {}
    for name, synth in gens.items():
        res[name] = evaluate(synth, sid, Rm, dom, gt_bi, gt_tri, real_tup,
                             ref_ids, ref_user, real_ids, real_user, grng)
        res[name]["n_states"] = len(symbols)
    return res


def main():
    import pickle
    torch.set_num_threads(1)
    print("[E3] start", flush=True)
    cache = os.path.join(OUT, "geolife_sess_cache.pkl")
    if os.path.exists(cache):
        print("[E3] reuse cached sess", flush=True)
        with open(cache, "rb") as f:
            sess, users = pickle.load(f)
    else:
        sem = load()
        sess = {t: (g["sig2"].tolist(), g["top1"].tolist(), g["user"].iloc[0])
                for t, g in sem.groupby("tid", sort=False)}
        users = np.array(sorted(sem.user.unique()))
        del sem                                      # free the 4M-row geodataframe before torch
        with open(cache, "wb") as f:
            pickle.dump((sess, users), f)
    U = len(users)
    ns = [n for n in NS if n <= U]
    print(f"[E3] U={U}, n={ns}, {R} nested paired replicates, Markov vs LSTM")
    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(users)
        for n in ns:
            res = run(sess, set(perm[:n]), np.random.default_rng(20_000 * r + n), tseed=1000 * r + n)
            for g in ("Markov", "LSTM"):
                rows.append({"rep": r, "n": n, "gen": g, **res[g]})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "geolife_e3_raw.csv"), index=False)

    mets = ["sem_biTV", "sem_triTV", "near_pct", "exact_copies", "DCR_synth", "DCR_base", "dDCR", "n_states"]
    agg = []
    for n in ns:
        for g in ("Markov", "LSTM"):
            sub = raw[(raw.n == n) & (raw.gen == g)]; row = {"n": n, "gen": g}
            for m in mets:
                a = sub[m].to_numpy(float)
                row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
            agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "geolife_e3_summary.csv"), index=False)
    print("\n=== E3 summary (mean +/- 95% CI) ===")
    print(adf[["n", "gen", "sem_biTV", "sem_triTV", "near_pct", "exact_copies", "dDCR"]].to_string(index=False))

    # paired Markov-LSTM deltas + sign-flip permutation test on sem_biTV and near_pct
    print("\n=== paired Markov vs LSTM (per replicate) ===")
    for n in ns:
        for m in ("sem_biTV", "near_pct"):
            mk = raw[(raw.n == n) & (raw.gen == "Markov")].sort_values("rep")[m].to_numpy()
            ls = raw[(raw.n == n) & (raw.gen == "LSTM")].sort_values("rep")[m].to_numpy()
            d = mk - ls; obs = d.mean()
            prng = np.random.default_rng(7)
            null = np.array([(d * prng.choice([-1, 1], len(d))).mean() for _ in range(10000)])
            p = float((np.abs(null) >= abs(obs)).mean())
            print(f"  n={n:3d} {m:9s}: Markov-LSTM = {obs:+.4f}  p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.array(ns); fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    C = {"Markov": "#2c7fb8", "LSTM": "#d95f0e"}
    for g in ("Markov", "LSTM"):
        s = adf[adf.gen == g]
        ax[0].errorbar(x, s.sem_biTV, yerr=s.sem_biTV_ci, fmt="o-", color=C[g], capsize=3, lw=2, label=g)
        ax[1].errorbar(x, s.near_pct, yerr=s.near_pct_ci, fmt="s-", color=C[g], capsize=3, lw=2, label=g)
        ax[2].errorbar(x, s.dDCR, yerr=s.dDCR_ci, fmt="D-", color=C[g], capsize=3, lw=2, label=g)
    ax[0].set_title("(A) Fidelity"); ax[0].set_ylabel("semantic generation TV (↓)")
    ax[1].set_title("(B) Near-copy disclosure"); ax[1].set_ylabel("% synth with max MUITAS ≥ 0.95 (↓)")
    ax[2].set_title("(C) ΔDCR (↑ safer)"); ax[2].set_ylabel("DCR synth − real baseline"); ax[2].axhline(0, color="#ccc", lw=.8)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("number of individuals  n"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off(); a.legend(fontsize=9)
    fig.suptitle(f"E3 GeoLife · generator capacity (Markov vs LSTM) · same MAT-Sum representation · {R} nested paired replicates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "geolife_e3.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
