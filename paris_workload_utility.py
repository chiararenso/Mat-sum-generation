"""
Query-workload utility on Paris-581: do the synthetic data answer the planner-style questions that
motivate the paper the way the real data do?

Place types are grouped into 8 functional categories (CATS below). Six questions are asked of the
synthetic release (N=200 sequences generated from n training individuals, MAT-Sum m=2,k=5) and of
the n real individuals; each question has one error measure (lower = better):
  Q1 where people go      : TV between category visit shares
  Q2 how they move        : TV between category-to-category flow shares (8x8, consecutive visits)
  Q3 common routines      : error (pp) on the 20 most frequent real category trigrams + recall@20
  Q4 who visits what      : mean abs. error (pp) of the share of individuals with >=1 visit per category
  Q5 when                 : JSD between the category x hour-of-day demand distributions
  Q6 how long they stay   : mean relative error of the median dwell time per category
Reference noise floor: the same questions asked of two disjoint random halves of the real
individuals ("real-vs-real"): the error one would see even with perfect data.

Same nested paired samples as RQ2/RQ3 (permutation seed = replicate); generators: order-1 Markov
(the release), order-2 Markov (light / heavy back-off). Dwell times are drawn from the empirical
per-state distribution; start times are bootstrapped from the real first-visit times (as in Sec. 5.3).
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matsum_summarize as MS
from val_rq3 import markov
from val_rq3_order2 import markov2

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, MAXLEN, N_SYN, R = 5, 150, 200, 20
NS = [20, 50, 100, 300, 581]
try:
    from scipy.stats import t as _t, wilcoxon
    TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093

CATS = {
    "transit": ["bus", "train", "tram", "subway", "rail", "railway", "transportation", "ferry", "stop_position",
                "bus_stop", "station", "aeroway"],
    "streets": ["bicycle_roads", "pedestrian_roads", "motor_roads", "square", "barrier"],
    "home": ["accomodation", "residential", "neighbourhood", "hamlet", "suburb", "quarter"],
    "commerce": ["commercial", "retail", "clothing_shoes_accessories", "generic_shop",
                 "general_store_department_store_mall", "stationery_gifts_books_newspapers",
                 "yourself_household_building_materials_gardening", "do", "it", "health_and_beauty",
                 "financial", "amenity", "office", "industrial"],
    "food": ["sustenance", "food_and_beverages"],
    "leisure": ["leisure", "sport", "outdoors_and_sport_vehicles", "attraction", "historic_attraction",
                "entertainment_art_and_culture", "art_music_hobbies", "religious"],
    "green": ["natural", "water", "meadow", "farmland", "orchard", "greenhouse_horticulture", "greenfield",
              "vineyard", "islet"],
    "services": ["facilities", "education", "healthcare"],
}
CAT_NAMES = list(CATS)
NC = len(CAT_NAMES)
LAB2CAT = {l: i for i, c in enumerate(CAT_NAMES) for l in CATS[c]}
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def runs(codes):
    if len(codes) == 0:
        return np.array([], int)
    return np.concatenate([[0], np.flatnonzero(np.diff(codes) != 0) + 1])


# ---------------------------------------------------------------- statistics of a set of sequences
def stats(seqs):
    """seqs: list of (cat_codes, dwell_s, hour). Returns the aggregates the six questions need."""
    cnt = np.zeros(NC); flow = np.zeros((NC, NC)); reach = np.zeros(NC); tod = np.zeros((NC, 24))
    tri = Counter(); dm, dd = [], []
    for mac, dw, hr in seqs:
        if len(mac) == 0:
            continue
        b = np.bincount(mac, minlength=NC); cnt += b; reach += (b > 0)
        if len(mac) > 1:
            np.add.at(flow, (mac[:-1], mac[1:]), 1)
        np.add.at(tod, (mac, hr), 1)
        for i in range(len(mac) - 2):
            tri[(mac[i], mac[i + 1], mac[i + 2])] += 1
        dm.append(mac); dd.append(dw)
    mac_all = np.concatenate(dm) if dm else np.array([], int)
    dw_all = np.concatenate(dd) if dd else np.array([], float)
    med = np.array([np.median(dw_all[mac_all == c]) if (mac_all == c).sum() >= 20 else np.nan for c in range(NC)])
    return dict(cnt=cnt, flow=flow, reach=reach / max(len(seqs), 1), tod=tod, tri=tri, med=med)


def tv(a, b):
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    return 0.5 * np.abs(a - b).sum()


def jsd(a, b):
    a = a.ravel() / max(a.sum(), 1e-12); b = b.ravel() / max(b.sum(), 1e-12); m = 0.5 * (a + b)
    kl = lambda p, q: np.sum(np.where(p > 0, p * np.log2(np.where(q > 0, p / np.where(q > 0, q, 1), 1)), 0.0))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def compare(X, Y):
    """Error of statistics X (synthetic or half A) against reference Y (real)."""
    out = {"Q1_category_share_TV": tv(X["cnt"], Y["cnt"]), "Q2_flow_TV": tv(X["flow"].ravel(), Y["flow"].ravel())}
    top = [m for m, _ in Y["tri"].most_common(20)]
    nx = max(sum(X["tri"].values()), 1); ny = max(sum(Y["tri"].values()), 1)
    out["Q3_motif_MAE_pp"] = 100 * float(np.mean([abs(X["tri"].get(m, 0) / nx - Y["tri"][m] / ny) for m in top]))
    topx = {m for m, _ in X["tri"].most_common(20)}
    out["Q3_motif_recall20"] = len(topx & set(top)) / max(len(top), 1)
    out["Q4_reach_MAE_pp"] = 100 * float(np.mean(np.abs(X["reach"] - Y["reach"])))
    out["Q5_time_of_day_JSD"] = jsd(X["tod"], Y["tod"])
    ok = ~np.isnan(X["med"]) & ~np.isnan(Y["med"]) & (Y["med"] > 0)
    out["Q6_dwell_relerr"] = float(np.mean(np.abs(X["med"][ok] - Y["med"][ok]) / Y["med"][ok])) if ok.any() else np.nan
    return out


# ---------------------------------------------------------------- data
def main():
    print("[load] Paris-581", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"]).reset_index(drop=True)
    top1 = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
    sig2 = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    sigc, uniq = pd.factorize(sig2); uniq = list(uniq)
    labc, labnames = pd.factorize(top1); labnames = list(labnames)
    missing = [l for l in labnames if l not in LAB2CAT]
    assert not missing, f"labels without category: {missing}"
    cat_of_lab = np.array([LAB2CAT[l] for l in labnames])
    mac = cat_of_lab[labc]
    loc = sem["time"].dt.tz_convert("Europe/Paris")
    hour = loc.dt.hour.to_numpy()
    off = (loc.dt.tz_localize(None) - sem["time"].dt.tz_localize(None)).dt.total_seconds().to_numpy()
    t0 = (sem["time"].astype("int64").to_numpy() // 10 ** 6).astype(np.int64)
    dur = np.nan_to_num(sem["duration"].to_numpy(float), nan=0.0)
    tid_codes, tids = pd.factorize(sem["tid"])
    bounds = np.flatnonzero(np.diff(tid_codes) != 0) + 1
    sl = list(zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(sem)]])))
    tids = np.array(tids)
    IDX = {t: s for t, s in zip(tids, sl)}
    print(f"  rows={len(sem)} individuals={len(tids)} labels={len(labnames)} states(raw)={len(uniq)}", flush=True)

    # per-individual caches: real reference at category level, raw signature runs
    real_ref, raw_runs = {}, {}
    for t, (a, b) in IDX.items():
        st = runs(mac[a:b]); mc = mac[a:b][st]
        real_ref[t] = (mc, np.add.reduceat(dur[a:b], st), hour[a:b][st])
        sr = runs(sigc[a:b]); raw_runs[t] = sigc[a:b][sr]
    first_start = np.array([(t0[IDX[t][0]], off[IDX[t][0]]) for t in tids])

    def synth_to_seq(states, dwell, start_epoch, start_off, dom_cat):
        mc = np.array([dom_cat[s] for s in states], int)
        cum = np.concatenate([[0.0], np.cumsum(dwell)[:-1]])
        hr = (((start_epoch + start_off + cum) // 3600) % 24).astype(int)
        st = runs(mc)
        return mc[st], np.add.reduceat(np.asarray(dwell, float), st), hr[st]

    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(tids)
        for n in NS:
            sample = list(perm[:n])
            cnts = Counter(int(c) for t in sample for c in raw_runs[t])
            freq = [c for c, k in cnts.items() if k >= K and len(uniq[c]) > 0] or [max(cnts, key=cnts.get)]
            fset = set(freq)
            lut = np.arange(len(uniq))
            for c in cnts:
                if c not in fset:
                    lut[c] = max(freq, key=lambda f: jac(uniq[c], uniq[f]))
            # state-level real runs (remapped), dwell lists, dominant label per state
            real_states, dwell_lists = [], defaultdict(list)
            lab_counts = defaultdict(lambda: np.zeros(len(labnames)))
            for t in sample:
                a, b = IDX[t]
                rc = lut[sigc[a:b]]; st = runs(rc)
                dw = np.add.reduceat(dur[a:b], st); sc = rc[st]
                for c, d in zip(sc, dw):
                    dwell_lists[uniq[c]].append(float(d))
                lab_counts_arr = np.bincount(rc * len(labnames) + labc[a:b], minlength=len(uniq) * len(labnames))
                nz = np.flatnonzero(lab_counts_arr)
                for z in nz:
                    lab_counts[uniq[z // len(labnames)]][z % len(labnames)] += lab_counts_arr[z]
                real_states.append([uniq[c] for c in sc[:MAXLEN]])
            dom_cat = {s: int(cat_of_lab[int(np.argmax(v))]) for s, v in lab_counts.items()}
            real_seqs = [real_ref[t] for t in sample]
            Ystats = stats(real_seqs)

            # real-vs-real floor: two disjoint halves of the n individuals
            frng = np.random.default_rng(7000 + 31 * r + n)
            pp = frng.permutation(len(sample)); h = len(sample) // 2
            fl = compare(stats([real_seqs[i] for i in pp[:h]]), stats([real_seqs[i] for i in pp[h:2 * h]]))
            rows.append({"rep": r, "n": n, "method": "real-vs-real", **fl})

            for k_, (name, gen) in enumerate([("Markov-1", None), ("Markov-2 light", None), ("Markov-2 heavy", "heavy")]):
                if name == "Markov-1":
                    synth = markov(real_states, N_SYN, np.random.default_rng(10_000 * r + n))
                else:
                    synth = markov2(real_states, N_SYN, np.random.default_rng(20_000 * r + n + 7 * k_), gen)
                srng = np.random.default_rng(30_000 * r + n + 11 * k_)
                seqs = []
                for s in synth:
                    if not s:
                        continue
                    dw = [float(srng.choice(dwell_lists[x])) if dwell_lists[x] else 0.0 for x in s]
                    j = srng.integers(len(first_start))
                    seqs.append(synth_to_seq(s, dw, first_start[j, 0], first_start[j, 1], dom_cat))
                rows.append({"rep": r, "n": n, "method": name, **compare(stats(seqs), Ystats)})
        print(f"  replicate {r + 1}/{R} done", flush=True)

    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "paris_workload_utility_raw.csv"), index=False)
    metrics = [c for c in raw.columns if c.startswith("Q")]
    agg = []
    for (n, m), g in raw.groupby(["n", "method"]):
        row = {"n": n, "method": m}
        for c in metrics:
            a = g[c].to_numpy(float); a = a[~np.isnan(a)]
            row[c] = float(a.mean()); row[c + "_ci"] = float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan
        agg.append(row)
    adf = pd.DataFrame(agg).sort_values(["n", "method"]); adf.to_csv(os.path.join(OUT, "paris_workload_utility_summary.csv"), index=False)

    pd.set_option("display.width", 250)
    for n in (100, 581):
        print(f"\n=== workload errors, n={n} (mean ± 95% CI over {R} replicates; lower better, except recall) ===")
        t = adf[adf.n == n].set_index("method")
        print(pd.DataFrame({c: t[c].map("{:.3f}".format) + "±" + t[c + "_ci"].map("{:.3f}".format) for c in metrics}).T.to_string())

    print("\n=== paired difference Markov-2 light minus Markov-1 (negative = order-2 better; recall: positive better) ===")
    prow = []
    for n in NS:
        a = raw[(raw.n == n) & (raw.method == "Markov-2 light")].sort_values("rep")
        b = raw[(raw.n == n) & (raw.method == "Markov-1")].sort_values("rep")
        for c in metrics:
            d = a[c].to_numpy(float) - b[c].to_numpy(float)
            d = d[~np.isnan(d)]
            p = float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else 1.0
            prow.append({"n": n, "metric": c, "d_mean": float(d.mean()), "wilcoxon_p": p})
    pdf = pd.DataFrame(prow); pdf.to_csv(os.path.join(OUT, "paris_workload_utility_paired.csv"), index=False)
    print(pdf.pivot(index="metric", columns="n", values="d_mean").round(4).to_string())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    titles = {"Q1_category_share_TV": "Q1 where people go (TV)", "Q2_flow_TV": "Q2 how they move (TV)",
              "Q3_motif_MAE_pp": "Q3 common routines (error, pp)", "Q4_reach_MAE_pp": "Q4 who visits what (error, pp)",
              "Q5_time_of_day_JSD": "Q5 when (JSD)", "Q6_dwell_relerr": "Q6 how long (rel. error)"}
    col = {"Markov-1": "#2c7fb8", "Markov-2 light": "#1b9e77", "Markov-2 heavy": "#7fc97f", "real-vs-real": "#555555"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2)); x = np.array(NS)
    for ax, c in zip(axes.ravel(), titles):
        for m in ["real-vs-real", "Markov-1", "Markov-2 light", "Markov-2 heavy"]:
            t = adf[adf.method == m].sort_values("n")
            ax.errorbar(t.n, t[c], yerr=t[c + "_ci"], fmt=("o--" if m == "real-vs-real" else "o-"), color=col[m],
                        capsize=3, lw=2, label=m)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x); ax.minorticks_off()
        ax.set_title(titles[c], fontsize=11); ax.set_xlabel("number of individuals n")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Query-workload utility on Paris-OSM: error of synthetic vs real answers (lower is better; "
                 "dashed = real-vs-real noise floor)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "paris_workload_utility.png"), dpi=130); plt.close(fig)
    print("saved summary, paired, raw CSVs and figure")


if __name__ == "__main__":
    main()
