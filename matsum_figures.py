"""STAGE 3b - static figures for the report / quick visual verification."""
import os
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.environ.get("MATSUM_OUT", os.path.join(HERE, "output"))
FIG  = os.environ.get("MATSUM_FIG", os.path.join(HERE, "figures")); os.makedirs(FIG, exist_ok=True)


def fig_trajectory(tid):
    gdf   = gpd.read_parquet(os.path.join(OUT, "trajectories.parquet"))
    tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()
    tg = gdf[gdf.tid == tid]
    visited = gpd.sjoin(tiles[["locationID", "category", "geometry"]],
                        tg[["geometry"]], predicate="contains").drop_duplicates("locationID")
    cats = sorted(visited["category"].unique())
    cmap = colormaps["tab20"]
    cidx = {c: cmap(i % 20) for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for _, t in visited.iterrows():
        gpd.GeoSeries([t.geometry]).plot(ax=ax, color=cidx[t["category"]],
                                         edgecolor="white", linewidth=0.4, alpha=0.75)
    ax.plot(tg.lon, tg.lat, color="#222", lw=1.2, alpha=0.7, zorder=5)
    ax.scatter(tg.lon, tg.lat, s=6, color="black", zorder=6)
    ax.set_title(f"Trajectory {tid[:8]} — {len(tg)} GPS points, "
                 f"{len(visited)} traversed cells, {len(cats)} distinct semantic contexts")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect("equal")
    fig.tight_layout(); p = os.path.join(FIG, f"verify_traj_{tid[:8]}.png")
    fig.savefig(p, dpi=130); plt.close(fig); print("  ", p)


def fig_srate_bar():
    r = pd.read_csv(os.path.join(OUT, "summarization_rate.csv"), index_col=0).sort_values("S_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([t[:8] for t in r.index], r["S_rate"], color="#2c7fb8")
    ax.axvline(r["S_rate"].mean(), color="#d95f0e", ls="--",
               label=f"mean = {r['S_rate'].mean():.3f}")
    ax.set_xlabel("S_rate  (1 - reps / points)"); ax.set_xlim(0, 1)
    ax.set_title("MAT-SUM summarization rate per trajectory (squares res17, τ=0.9)")
    ax.legend(); fig.tight_layout()
    p = os.path.join(FIG, "srate_per_trajectory.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  ", p)


def fig_tau_sweep():
    s = pd.read_csv(os.path.join(OUT, "tau_sweep.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s["tau"], s["S_rate"], "o-", color="#2c7fb8", lw=2)
    for _, row in s.iterrows():
        ax.annotate(f"{row['S_rate']:.3f}", (row["tau"], row["S_rate"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("similarity threshold τ"); ax.set_ylabel("dataset S_rate")
    ax.invert_xaxis(); ax.set_ylim(0.7, 1.0)
    ax.set_title("Summarization rate vs τ  (Paris / Akkodis, squares res17)")
    ax.grid(alpha=0.3); fig.tight_layout()
    p = os.path.join(FIG, "tau_sweep.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  ", p)


def fig_tradeoff():
    s = pd.read_csv(os.path.join(OUT, "evaluation_srate_muitas.csv"))
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(s["tau"], s["S_rate2"], "o-", color="#2c7fb8", lw=2, label="S_rate (compression)")
    ax1.set_xlabel("similarity threshold τ"); ax1.set_ylabel("S_rate", color="#2c7fb8")
    ax1.invert_xaxis(); ax1.set_ylim(0.6, 1.02); ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(s["tau"], s["MUITAS"], "s--", color="#d95f0e", lw=2, label="MUITAS (semantic quality)")
    ax2.set_ylabel("MUITAS", color="#d95f0e"); ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis="y", labelcolor="#d95f0e")
    ax1.axvspan(0.9, 0.8, color="green", alpha=0.07)
    ax1.set_title("Compression vs semantic quality trade-off (Paris / Akkodis)\n"
                  "sweet spot τ≈0.8–0.9 shaded")
    fig.tight_layout()
    p = os.path.join(FIG, "tradeoff_srate_muitas.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  ", p)


if __name__ == "__main__":
    print("[figures]")
    fig_trajectory("2de4901f-f621-5a67-a327-467d34c4f58a")  # longest trajectory
    fig_srate_bar()
    fig_tau_sweep()
    fig_tradeoff()
