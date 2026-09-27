"""
Shokri-style shadow-model membership-inference attack on Paris-581 (fully specified, addressing
the reviewer request for exact reproducibility parameters -- see paper Sec. 5.5).

Setup (all parameters explicit):
  - Population: the same 581 individuals used throughout the paper. A fixed 20% (116) are
    reserved as the TARGET non-member split (same split as paris_mia.py, rng seed 2024); the
    remaining 465 are the TARGET member pool.
  - TARGET model: a single order-1 Markov model (m=2,k=5 MAT-Sum vocabulary) trained on all 465
    target-member individuals; releases N_SYN=200 synthetic sequences (the paper's operating
    point). Evaluated once per outer repeat.
  - SHADOW models: K_SHADOW=20 shadows per outer repeat. Each shadow draws its OWN fresh random
    465/116 partition of the full 581-person population (independent of, and never equal to, the
    target's fixed split), trains a Markov model on its 465 "in" individuals, and releases its own
    N_SYN=200 synthetic sequences.
  - Features per (shadow or target, individual): (closeness = max MUITAS to the release's
    synthetic set, real sequence length). Label = 1 if that individual was in that model's
    training set, else 0.
  - Attack classifier: scikit-learn LogisticRegression on StandardScaler-normalised 2D features,
    trained ONCE per outer repeat on the POOLED (individual, shadow) pairs from all K_SHADOW
    shadows (581 individuals x 20 shadows = 11,620 labelled examples per repeat).
  - Evaluation: the trained classifier scores all 581 individuals' features against the TARGET's
    own synthetic release (never used to train any shadow or the classifier); ROC-AUC against the
    true target member/non-member labels.
  - We also compute the simple black-box distance attack (closeness alone, no classifier) on the
    SAME target release/split, for the paired comparison already reported in the paper.
  - R=10 outer repeats (fresh shadows + fresh target draw each repeat) for mean +/- 95% CI.
"""
import os, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matsum_summarize as MS
from val_rq3 import rle, subset_matrix, sim, markov

OUT = os.environ["MATSUM_OUT"]; FIG = os.environ.get("MATSUM_FIG", OUT)
K, N_SYN = 5, 200
K_SHADOW, R, TEST_FRAC = 20, 10, 0.20
jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 0.0
try:
    from scipy.stats import t as _t, wilcoxon
    TCRIT = float(_t.ppf(0.975, R - 1))
except Exception:
    TCRIT = 2.262


def build_model_and_release(tids_in, per, grng):
    """Fit the m=2,k=5 MAT-Sum vocabulary + order-1 Markov on tids_in, release N_SYN synthetic
    sequences. Returns (sid, Rm, synth_ids) needed to score ANY individual's closeness to this
    release."""
    raw = {t: rle(per[t]) for t in tids_in}
    fs = Counter(s for t in tids_in for s in raw[t])
    freq = [s for s, c in fs.items() if c >= K and len(s) > 0] or [fs.most_common(1)[0][0]]
    freq_set = set(freq)
    remap = {s: (s if s in freq_set else max(freq, key=lambda f: jac(s, f))) for s in fs}
    real = [rle([remap[s] for s in raw[t]]) for t in tids_in]
    symbols = sorted({s for v in real for s in v}, key=lambda z: sorted(z))
    sid = {s: i for i, s in enumerate(symbols)}
    Rm = subset_matrix(symbols)
    synth = markov(real, N_SYN, grng)
    synth_ids = [np.array([sid[s] for s in v]) for v in synth]
    return freq_set, freq, sid, Rm, synth_ids


def map_ids(raw_sig2, freq_set, freq_list, sid):
    seq = [(s if s in freq_set else max(freq_list, key=lambda f: jac(s, f))) for s in rle(raw_sig2)]
    return np.array([sid[s] for s in rle(seq)])


def features_for_all(all_tids, per, freq_set, freq, sid, Rm, synth_ids):
    """closeness (max MUITAS to synth_ids) and real sequence length, for every individual."""
    close, length = [], []
    for t in all_tids:
        ids = map_ids(per[t], freq_set, freq, sid)
        length.append(len(ids))
        close.append(max((sim(Rm, ids, z) for z in synth_ids), default=0.0))
    return np.array(close), np.array(length, float)


def main():
    print("[load] Paris-581", flush=True)
    gdf, areas = MS.load()
    sem = MS.semantic_mapping(gdf, areas).sort_values(["tid", "time"])
    sem["sig2"] = sem["label_tfidf"].apply(lambda x: frozenset(list(x)[:2]) if hasattr(x, "__len__") else frozenset())
    per = {t: list(g["sig2"]) for t, g in sem.groupby("tid", sort=False)}
    all_tids = np.array(list(per.keys()))
    U = len(all_tids)

    rng0 = np.random.default_rng(2024)
    perm0 = rng0.permutation(all_tids)
    n_test = int(TEST_FRAC * U)
    target_nonmember = set(perm0[:n_test]); target_pool = perm0[n_test:]
    n_target_train = len(target_pool)
    print(f"  population={U}  target member pool={n_target_train}  target non-members={len(target_nonmember)}", flush=True)
    print(f"  shadows: K_SHADOW={K_SHADOW}, each with its own random {n_target_train}/{len(target_nonmember)} split", flush=True)

    rows = []
    for r in range(R):
        # ---- TARGET model + release (fixed split, never used by any shadow) ----
        target_grng = np.random.default_rng(900_000 + r)
        t_freq_set, t_freq, t_sid, t_Rm, t_synth_ids = build_model_and_release(list(target_pool), per, target_grng)
        target_labels = {t: int(t in set(target_pool)) for t in all_tids}
        t_close, t_len = features_for_all(all_tids, per, t_freq_set, t_freq, t_sid, t_Rm, t_synth_ids)
        y_target = np.array([target_labels[t] for t in all_tids])

        # simple black-box attack on this same target release
        auc_simple = roc_auc_score(y_target, t_close)

        # ---- SHADOWS: fresh random splits of the SAME population, independent of target's ----
        X_shadow, y_shadow = [], []
        for k in range(K_SHADOW):
            shadow_rng = np.random.default_rng(10_000 * r + k)
            shadow_perm = shadow_rng.permutation(all_tids)
            shadow_train = set(shadow_perm[:n_target_train])
            gen_rng = np.random.default_rng(500_000 * r + 1000 * k)
            s_freq_set, s_freq, s_sid, s_Rm, s_synth_ids = build_model_and_release(list(shadow_train), per, gen_rng)
            s_close, s_len = features_for_all(all_tids, per, s_freq_set, s_freq, s_sid, s_Rm, s_synth_ids)
            s_label = np.array([int(t in shadow_train) for t in all_tids])
            X_shadow.append(np.column_stack([s_close, s_len]))
            y_shadow.append(s_label)
        X_shadow = np.vstack(X_shadow); y_shadow = np.concatenate(y_shadow)

        # ---- attack classifier: pooled shadow data -> evaluate on target ----
        scaler = StandardScaler().fit(X_shadow)
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_shadow), y_shadow)
        X_target = np.column_stack([t_close, t_len])
        p_target = clf.predict_proba(scaler.transform(X_target))[:, 1]
        auc_shadow = roc_auc_score(y_target, p_target)

        rows.append({"rep": r, "auc_shadow": auc_shadow, "auc_simple": auc_simple,
                     "n_shadow_examples": len(y_shadow), "n_target_eval": len(y_target)})
        print(f"  repeat {r+1}/{R}: AUC shadow-classifier={auc_shadow:.4f}  simple black-box={auc_simple:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "paris_shadow_mia_raw.csv"), index=False)

    a_shadow = df.auc_shadow.to_numpy(); a_simple = df.auc_simple.to_numpy()
    d = a_shadow - a_simple
    ci_shadow = TCRIT * a_shadow.std(ddof=1) / np.sqrt(R)
    ci_simple = TCRIT * a_simple.std(ddof=1) / np.sqrt(R)
    ci_d = TCRIT * d.std(ddof=1) / np.sqrt(R)
    wp = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0

    print(f"\n=== Paris shadow-model MIA (K_SHADOW={K_SHADOW} per repeat, R={R} outer repeats) ===")
    print(f"  shadow-classifier AUC : {a_shadow.mean():.4f} ± {ci_shadow:.4f} (95% CI)")
    print(f"  simple black-box AUC  : {a_simple.mean():.4f} ± {ci_simple:.4f} (95% CI)")
    print(f"  paired diff (shadow-simple): {d.mean():+.4f} ± {ci_d:.4f} (95% CI), Wilcoxon p={wp:.3f}")

    summary = pd.DataFrame([{
        "K_SHADOW": K_SHADOW, "R": R,
        "n_target_train": n_target_train, "n_target_nonmember": len(target_nonmember),
        "auc_shadow_mean": round(float(a_shadow.mean()), 4), "auc_shadow_ci95": round(float(ci_shadow), 4),
        "auc_simple_mean": round(float(a_simple.mean()), 4), "auc_simple_ci95": round(float(ci_simple), 4),
        "paired_diff_mean": round(float(d.mean()), 4), "paired_diff_ci95": round(float(ci_d), 4),
        "wilcoxon_p": round(wp, 4),
    }])
    summary.to_csv(os.path.join(OUT, "paris_shadow_mia_summary.csv"), index=False)
    print("\nsaved:", os.path.join(OUT, "paris_shadow_mia_summary.csv"))


if __name__ == "__main__":
    main()
