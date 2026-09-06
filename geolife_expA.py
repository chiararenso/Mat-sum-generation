"""
Experiment A -- decoupling the number of individuals from the information per individual (GeoLife).
We grid over n = number of users (privacy units) and s = cap on sessions per user (info/individual),
holding the MAT-Sum+Markov pipeline fixed. Within a replicate the SAME n users are used across s, and
increasing n adds users, so the two structural knobs move independently.

Question: does generation QUALITY depend on the total transition support S (~ n*s), collapsing onto a
single curve, while DISCLOSURE depends on the number of individuals n (and/or the per-individual
richness s)? We measure semantic generation TV, exact copies, user-aware dDCR, near-copy rate, and
diagnostics (total sessions/transitions/states).
"""
import os, pickle, numpy as np, pandas as pd
from collections import Counter, defaultdict
from geolife_e1 import rle, bigram, tv, submat, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, N_SYN, R, REF, NEAR = 5, 200, 10, 400, 0.95
NS = [20, 50, 100, 150]
SS = [1, 2, 5, 10, 20]
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.262


def run(tids, sess, grng):
    raw = {t: rle(sess[t][0]) for t in tids}
    fs = Counter(s for t in tids for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids]
    ru = [sess[t][2] for t in tids]
    dom = defaultdict(Counter); gt = []
    for t in tids:
        for s, l in zip(sess[t][0], sess[t][1]):
            dom[remap[s]][l] += 1
        gt.append(rle(sess[t][1]))
    dom = {k2: c.most_common(1)[0][0] for k2, c in dom.items()}
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}; Rm = submat(symbols)
    real_ids = [np.array([sid[s] for s in v]) for v in real]
    gt_bi = bigram(gt)

    synth = markov(real, N_SYN, grng)
    synth_ids = [np.array([sid[s] for s in v]) for v in synth]
    synth_lab = [rle([dom[s] for s in v]) for v in synth]
    sem = tv(bigram(synth_lab), gt_bi)
    real_tup = set(tuple(v) for v in real); exact = sum(tuple(v) in real_tup for v in synth)

    ridx = grng.choice(len(real_ids), min(REF, len(real_ids)), replace=False)
    ref = [real_ids[i] for i in ridx]; ref_u = [ru[i] for i in ridx]
    maxsim = [max((sim(Rm, s, r) for r in ref), default=0.0) for s in synth_ids]
    dcr_s = float(np.mean([1 - m for m in maxsim]))
    near = 100.0 * float(np.mean([m >= NEAR for m in maxsim]))
    qidx = grng.choice(len(real_ids), min(REF, len(real_ids)), replace=False)
    per = defaultdict(list)
    for qi in qidx:
        u = ru[qi]
        best = max((sim(Rm, real_ids[qi], ref[j]) for j in range(len(ref)) if ref_u[j] != u), default=0.0)
        per[u].append(1 - best)
    dcr_b = float(np.mean([np.mean(v) for v in per.values()])) if per else 0.0

    supp = np.array(list(Counter(s for v in real for s in v).values()))
    return {"sem_genTV": sem, "exact": exact, "DCR_synth": dcr_s, "DCR_base": dcr_b, "dDCR": dcr_s - dcr_b,
            "near_pct": near, "sessions": len(tids), "transitions": int(sum(max(0, len(v) - 1) for v in real)),
            "n_states": len(symbols), "median_support": float(np.median(supp)) if len(supp) else 0.0}


def main():
    with open(os.path.join(OUT, "geolife_sess_cache.pkl"), "rb") as f:
        sess, _ = pickle.load(f)
    u2t = defaultdict(list)
    for t, (_, _, u) in sess.items():
        u2t[u].append(t)
    users = np.array(sorted(u2t)); U = len(users)
    print(f"[expA] U={U}, grid n={NS} x s={SS}, R={R}", flush=True)

    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(users)
        for n in NS:
            sel = perm[:n]
            for s in SS:
                grng = np.random.default_rng(70_000 * r + 100 * n + s)
                tids = []
                for u in sel:
                    ut = u2t[u]
                    idx = grng.choice(len(ut), min(s, len(ut)), replace=False)
                    tids += [ut[i] for i in idx]
                res = run(tids, sess, grng)
                rows.append({"rep": r, "n": n, "s": s, **res})
        print(f"  replicate {r+1}/{R} done", flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "geolife_expA_raw.csv"), index=False)

    mets = ["sem_genTV", "exact", "dDCR", "DCR_synth", "DCR_base", "near_pct", "sessions", "transitions",
            "n_states", "median_support"]
    agg = []
    for n in NS:
        for s in SS:
            sub = raw[(raw.n == n) & (raw.s == s)]; row = {"n": n, "s": s}
            for m in mets:
                a = sub[m].to_numpy(float)
                row[m] = round(float(a.mean()), 4); row[m + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
            agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "geolife_expA_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== expA summary ===")
    print(adf[["n", "s", "sessions", "transitions", "n_states", "sem_genTV", "near_pct", "dDCR"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def heat(ax, col, title, cmap):
        M = np.array([[adf[(adf.n == n) & (adf.s == s)][col].iloc[0] for n in NS] for s in SS])
        im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(NS))); ax.set_xticklabels(NS); ax.set_yticks(range(len(SS))); ax.set_yticklabels(SS)
        ax.set_xlabel("n  (individuals)"); ax.set_ylabel("s  (sessions/user cap)"); ax.set_title(title)
        for i in range(len(SS)):
            for j in range(len(NS)):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmap == "viridis_r" else "black")
        plt.colorbar(im, ax=ax, shrink=.85)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    heat(ax[0], "sem_genTV", "(A) Quality: semantic generation TV (↓)", "viridis_r")
    heat(ax[1], "dDCR", "(B) Disclosure: ΔDCR (↑ safer)", "RdYlGn")
    fig.suptitle("Experiment A · decoupling #individuals (n) from info-per-individual (s) · GeoLife", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, "fig_expA_heat.png"), dpi=130); plt.close(fig)

    # collapse: quality vs total sessions; disclosure vs n
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(NS)))
    for c, n in zip(cmap, NS):
        d = adf[adf.n == n].sort_values("sessions")
        ax[0].errorbar(d.sessions, d.sem_genTV, yerr=d.sem_genTV_ci, fmt="o-", color=c, capsize=3, lw=2, label=f"n={n}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("total training sessions  (≈ transition support S)")
    ax[0].set_ylabel("semantic generation TV (↓)")
    ax[0].set_title("(A) Quality collapses onto total support"); ax[0].legend(fontsize=8, title="fixed n, vary s")
    smap = plt.cm.viridis(np.linspace(0.15, 0.85, len(SS)))
    for c, s in zip(smap, SS):
        d = adf[adf.s == s].sort_values("n")
        ax[1].errorbar(d.n, d.dDCR, yerr=d.dDCR_ci, fmt="s-", color=c, capsize=3, lw=2, label=f"s={s}")
    ax[1].axhline(0, color="#999", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_xticks(NS); ax[1].set_xticklabels(NS); ax[1].minorticks_off()
    ax[1].set_xlabel("n  (number of individuals)"); ax[1].set_ylabel("ΔDCR (↑ safer)")
    ax[1].set_title("(B) Disclosure vs number of individuals"); ax[1].legend(fontsize=8, title="fixed s, vary n")
    fig.suptitle("Experiment A · quality ~ total support S,  disclosure ~ structure (GeoLife)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(FIG, "fig_expA_collapse.png"), dpi=130); plt.close(fig)
    print("\nfigures: fig_expA_heat.png, fig_expA_collapse.png")


if __name__ == "__main__":
    main()
