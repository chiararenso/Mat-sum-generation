"""
E1 -- role of the abstraction on GeoLife (external replication of RQ1).
Unit = user; sequence = session (NO cross-session transitions). Sample 100 users per replicate
(20 paired replicates); Grid / Cluster / MAT-Sum get the SAME users and sessions. Per replicate we
build MAT-Sum (m=2,k=5), record its effective #states K_r, and match Grid (visited cells ~K_r) and
Cluster (KMeans K_r) to the same K_r; identical order-1 Markov and session-length model; N=200
synthetic sessions/method. Points are down-sampled per session (every DS-th) to bound compute; the
map/aspects are the fixed public Beijing enrichment; the MAT-Sum vocabulary is rebuilt per sample.

Metrics (mean +/- 95% CI over 20 paired replicates):
  native-state bigram TV; semantic abstraction distortion; semantic generation TV;
  exact session copies /200; near-copy rate (common-label MUITAS>=0.95).
"""
import os, numpy as np, pandas as pd, geopandas as gpd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

OUT = os.environ["MATSUM_OUT"]; GEO = os.environ["GEO_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
M, K, N_SYN, DS = 2, 5, 200, 4
NRQ1, R, REF_CAP, NEAR = 100, 20, 300, 0.95
MINV, MINS = 10, 3
rng = np.random.default_rng(0)
try:
    from scipy.stats import t as _t; TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.093
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0


def rle(x):
    out = []
    for v in x:
        if not out or out[-1] != v:
            out.append(v)
    return out


def bigram(seqs):
    c = Counter()
    for v in seqs:
        for a, b in zip(v, v[1:]):
            c[(a, b)] += 1
    n = sum(c.values()) or 1
    return {k: x / n for k, x in c.items()}


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def submat(symbols):
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
    START = "<S>"; tr = defaultdict(Counter); st = set()
    for v in real:
        p = START
        for s in v:
            tr[p][s] += 1; st.add(s); p = s
    st = list(st); L = [len(v) for v in real if len(v)] or [1]
    def nxt(s):
        c = np.array([tr[s][x] for x in st], float)
        return st[grng.choice(len(st), p=(c + .1) / (c.sum() + .1 * len(st)))]
    out = []
    for _ in range(n):
        ln = int(grng.choice(L)); s = START; seq = []
        for _ in range(ln):
            s = nxt(s); seq.append(s)
        out.append(seq)
    return out


def load():
    print("[load] GeoLife points + fixed Beijing enrichment; one-time semantic sjoin")
    P = pd.read_parquet(os.path.join(GEO, "geolife_points.parquet"))
    gdf = gpd.GeoDataFrame(P, geometry=gpd.points_from_xy(P.lon, P.lat), crs="EPSG:4326")
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    sem = gpd.sjoin(gdf[["tid", "user", "session", "time", "lat", "lon", "geometry"]],
                    tiles[["category", "label_tfidf", "geometry"]], predicate="within")
    sem = sem[~sem.index.duplicated(keep="first")].sort_values(["user", "session", "time"])
    # frozen validity filter (>=MINV semantic visits/session, >=MINS valid sessions/user)
    def nv(a):
        a = a.to_numpy(); return int(1 + (a[1:] != a[:-1]).sum()) if len(a) else 0
    vis = sem.groupby("tid")["category"].agg(nv)
    valid_tid = set(vis[vis >= MINV].index)
    sem = sem[sem.tid.isin(valid_tid)]
    vps = sem.groupby("user")["tid"].nunique()
    final_users = set(vps[vps >= MINS].index)
    sem = sem[sem.user.isin(final_users)]
    sem = sem.iloc[::DS].copy()                                   # down-sample to bound compute
    xy = gpd.GeoSeries(sem.geometry, crs="EPSG:4326").to_crs(2154)
    sem["x"], sem["y"] = xy.x.values, xy.y.values
    sem["sig2"] = sem["label_tfidf"].apply(lambda v: frozenset(list(v)[:2]) if hasattr(v, "__len__") else frozenset())
    sem["top1"] = sem["label_tfidf"].apply(lambda v: v[0] if hasattr(v, "__len__") and len(v) else "NA")
    print(f"  users={sem.user.nunique()} sessions={sem.tid.nunique()} points(ds)={len(sem)}")
    return sem


def sess_seqs(df, col):
    """per-session RLE state sequences (no cross-session)."""
    return [rle(g[col].tolist()) for _, g in df.groupby("tid", sort=False)]


def run(sem, users, grng):
    d = sem[sem.user.isin(users)].sort_values(["user", "session", "time"])
    # MAT-Sum per-sample vocabulary
    raw_v = [rle(g["sig2"].tolist()) for _, g in d.groupby("tid", sort=False)]
    fs = Counter(s for v in raw_v for s in v)
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    remap = {s: (s if s in freq else max(freq, key=lambda f: jac(s, f))) for s in fs}
    d = d.copy(); d["matsum"] = d["sig2"].map(lambda s: remap.get(s, s))
    Kr = d["matsum"].nunique()
    # Grid matched to ~Kr visited cells
    lo, hi = 0.0008, 0.05
    for _ in range(28):
        g = (lo + hi) / 2
        nv = len(set(zip((d.lat / g).round().astype(int), (d.lon / g).round().astype(int))))
        lo, hi = (g, hi) if nv > Kr else (lo, g)
    g = (lo + hi) / 2
    d["grid"] = list(zip((d.lat / g).round().astype(int), (d.lon / g).round().astype(int)))
    # Cluster matched to Kr
    XY = d[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=Kr, n_init=2, random_state=0).fit(XY[grng.choice(len(XY), min(30000, len(XY)), replace=False)])
    d["cluster"] = km.predict(XY)

    gt = sess_seqs(d, "top1"); gt_bi = bigram(gt)
    out = {}
    for name, col in [("Grid", "grid"), ("Cluster", "cluster"), ("MAT-Sum", "matsum")]:
        dom = d.groupby(col)["top1"].agg(lambda s: s.value_counts().index[0]).to_dict()
        real = sess_seqs(d, col)
        synth = markov(real, N_SYN, grng)
        real_lab = [rle([dom[s] for s in v]) for v in real]
        synth_lab = [rle([dom[s] for s in v]) for v in synth]
        # exact session copies
        real_tup = set(tuple(v) for v in real)
        exact = sum(tuple(v) in real_tup for v in synth)
        # near-copy in common-label space (singleton dominant-label sets)
        labs = sorted({l for v in gt for l in v} | {"NA"})
        lid = {l: i for i, l in enumerate(labs)}
        Rl = np.eye(len(labs), dtype=bool)                        # singleton subset == equality
        ref = [np.array([lid.get(l, 0) for l in s]) for s in
               [real_lab[i] for i in grng.choice(len(real_lab), min(REF_CAP, len(real_lab)), replace=False)]]
        near = 100.0 * np.mean([max((sim(Rl, np.array([lid.get(l, 0) for l in s]), r) for r in ref), default=0.0) >= NEAR
                                for s in synth_lab])
        out[name] = {"n_states": Kr, "native_biTV": tv(bigram(real), bigram(synth)),
                     "distortion": tv(bigram(real_lab), gt_bi), "sem_genTV": tv(bigram(synth_lab), gt_bi),
                     "exact_copies": exact, "near_pct": float(near)}
    return out


def main():
    sem = load()
    users = np.array(sorted(sem.user.unique()))
    n = min(NRQ1, int(0.8 * len(users)))
    print(f"[E1] n_RQ1={n}, U={len(users)}, {R} paired replicates")
    rows = []
    for r in range(R):
        perm = np.random.default_rng(r).permutation(users)
        res = run(sem, set(perm[:n]), np.random.default_rng(1000 + r))
        for m in ("Grid", "Cluster", "MAT-Sum"):
            rows.append({"rep": r, "method": m, **res[m]})
        print(f"  replicate {r+1}/{R} done (Kr~{res['MAT-Sum']['n_states']})", flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(OUT, "geolife_e1_raw.csv"), index=False)

    mets = ["native_biTV", "distortion", "sem_genTV", "exact_copies", "near_pct", "n_states"]
    agg = []
    for m in ("Grid", "Cluster", "MAT-Sum"):
        sub = raw[raw.method == m]; row = {"method": m}
        for k in mets:
            a = sub[k].to_numpy(float)
            row[k] = round(float(a.mean()), 4); row[k + "_ci"] = round(float(TCRIT * a.std(ddof=1) / np.sqrt(len(a))), 4)
        agg.append(row)
    adf = pd.DataFrame(agg); adf.to_csv(os.path.join(OUT, "geolife_e1_summary.csv"), index=False)
    print("\n=== E1 summary (mean +/- 95% CI) ===")
    print(adf[["method", "n_states", "native_biTV", "distortion", "sem_genTV", "exact_copies", "near_pct"]].to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(len(adf)); fig, (aA, aB) = plt.subplots(1, 2, figsize=(11, 4.7))
    aA.bar(x, adf.native_biTV, 0.55, yerr=adf.native_biTV_ci, capsize=4, color="#2c7fb8")
    aA.set_xticks(x); aA.set_xticklabels(adf.method); aA.set_ylabel("TV (↓ better)")
    aA.set_title("(A) native-state bigram TV")
    w = 0.38
    aB.bar(x - w/2, adf.distortion, w, yerr=adf.distortion_ci, capsize=4, color="#8aa0b3", label="abstraction distortion")
    aB.bar(x + w/2, adf.sem_genTV, w, yerr=adf.sem_genTV_ci, capsize=4, color="#d95f0e", label="generation (synth→labels)")
    aB.set_xticks(x); aB.set_xticklabels(adf.method); aB.set_title("(B) common-label-space TV"); aB.legend(fontsize=9)
    fig.suptitle(f"E1 GeoLife · role of the abstraction · ~{int(adf.n_states.mean())} states · n={n} · {R} paired samples (95% CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "geolife_e1.png"); os.makedirs(FIG, exist_ok=True)
    fig.savefig(p, dpi=130); plt.close(fig); print("\nfigure:", p)


if __name__ == "__main__":
    main()
