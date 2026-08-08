"""
Disclosure tail analysis on GeoLife (MAT-Sum + order-1 Markov), mirroring paris_disclosure.py.
Uses the cached sess (geolife_sess_cache.pkl). For each n over 20 nested paired replicates we generate
200 synthetic sessions and record each synthetic session's max MUITAS to a training reference (all
sessions of the sampled users, capped at REF_CAP for tractability). Baseline: each real session vs
sessions of OTHER users (user-aware leave-one-user-out max MUITAS). Reports near-copy(tau) and DCR tail.
"""
import os, pickle, numpy as np, pandas as pd
from collections import Counter, defaultdict
from geolife_e1 import rle, submat, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, N_SYN, R, REF_CAP = 5, 200, 20, 3000
NS = [10, 20, 50, 100, 152]
TAUS = np.round(np.arange(0.80, 1.0001, 0.02), 3)
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def build(users, sess):
    tids = [t for t in sess if sess[t][2] in users]
    raw = {t: rle(sess[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids]
    ru = [sess[t][2] for t in tids]
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = submat(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]
    return real, real_ids, ru, symbols, sid, Rm


def main():
    with open(os.path.join(OUT, "geolife_sess_cache.pkl"), "rb") as f:
        sess, users_all = pickle.load(f)
    users_all = np.array(sorted(users_all))
    print(f"[geolife disclosure] U={len(users_all)}", flush=True)
    store = {n: {"synth": [], "real": []} for n in NS}
    for r in range(R):
        perm = np.random.default_rng(r).permutation(users_all)
        for n in NS:
            real, real_ids, ru, symbols, sid, Rm = build(set(perm[:n]), sess)
            grng = np.random.default_rng(30_000 * r + n)
            # reference for synth (cap)
            ridx = grng.choice(len(real_ids), min(REF_CAP, len(real_ids)), replace=False)
            ref = [real_ids[i] for i in ridx]; ref_u = [ru[i] for i in ridx]
            synth = markov(real, N_SYN, grng)
            synth_ids = [np.array([sid[s] for s in v]) for v in synth]
            store[n]["synth"].extend(max((sim(Rm, s, x) for x in ref), default=0.0) for s in synth_ids)
            # baseline: sample up to 400 real queries vs ref of OTHER users
            qidx = grng.choice(len(real_ids), min(400, len(real_ids)), replace=False)
            store[n]["real"].extend(
                max((sim(Rm, real_ids[q], ref[j]) for j in range(len(ref)) if ref_u[j] != ru[q]), default=0.0)
                for q in qidx)
        print(f"  replicate {r+1}/{R} done", flush=True)

    rows = []
    for n in NS:
        syn = np.array(store[n]["synth"]); rel = np.array(store[n]["real"]); dcr = 1 - syn
        row = {"n": n, "DCR_mean": float(dcr.mean()), "DCR_min": float(dcr.min()),
               "DCR_p05": float(np.percentile(dcr, 5)),
               "pct_DCR_lt0.10": float(100 * (dcr < 0.10).mean()),
               "pct_DCR_lt0.05": float(100 * (dcr < 0.05).mean())}
        for tau in TAUS:
            row[f"near_synth@{tau}"] = float(100 * (syn >= tau).mean())
            row[f"near_real@{tau}"] = float(100 * (rel >= tau).mean())
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "geolife_disclosure.csv"), index=False)
    np.savez(os.path.join(OUT, "geolife_disclosure_raw.npz"),
             **{f"synth_{n}": np.array(store[n]["synth"]) for n in NS},
             **{f"real_{n}": np.array(store[n]["real"]) for n in NS})
    pd.set_option("display.width", 200)
    print("\n=== GeoLife DCR tail ===")
    print(df[["n", "DCR_mean", "DCR_p05", "DCR_min", "pct_DCR_lt0.10", "pct_DCR_lt0.05"]].to_string(index=False))
    print("\n=== near-copy(tau): synth vs real-real (user-aware) baseline (%) ===")
    show = ["n"] + [f"near_synth@{t}" for t in (0.9, 0.95, 1.0)] + [f"near_real@{t}" for t in (0.9, 0.95, 1.0)]
    print(df[show].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(NS)))
    for c, n in zip(cmap, NS):
        sub = df[df.n == n]
        ax[0].plot(TAUS, [sub[f"near_synth@{t}"].iloc[0] for t in TAUS], "-", color=c, lw=2, label=f"synth n={n}")
        ax[0].plot(TAUS, [sub[f"near_real@{t}"].iloc[0] for t in TAUS], "--", color=c, lw=1.2, alpha=.7)
    ax[0].set_xlabel("near-copy threshold τ (MUITAS)"); ax[0].set_ylabel("% records ≥ τ (↓ safer)")
    ax[0].set_title("(A) Near-copy rate vs threshold\nsolid = synth→train · dashed = real→real baseline")
    ax[0].legend(fontsize=8); ax[0].set_yscale("symlog", linthresh=1)
    ax[1].plot(df.n, df.DCR_mean, "o-", color="#d95f0e", lw=2, label="mean DCR")
    ax[1].plot(df.n, df.DCR_p05, "s--", color="#2c7fb8", lw=2, label="5th-percentile DCR (tail)")
    ax[1].plot(df.n, df.DCR_min, "^:", color="#555", lw=1.5, label="min DCR (worst case)")
    ax[1].set_xscale("log"); ax[1].set_xticks(NS); ax[1].set_xticklabels(NS); ax[1].minorticks_off()
    ax[1].set_xlabel("number of individuals  n"); ax[1].set_ylabel("DCR = 1 − max MUITAS (↑ safer)")
    ax[1].set_title("(B) DCR distribution: mean vs tail"); ax[1].legend(fontsize=9)
    fig.suptitle("GeoLife · disclosure tail analysis · MAT-Sum+Markov · 20 replicates", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(FIG, "geolife_disclosure.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
