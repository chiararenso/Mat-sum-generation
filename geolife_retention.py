"""GeoLife pre-E1 minimal audit: step-by-step retention table + 3 diagnostic plots.
Quality control that the frozen filters do not produce a deformed dataset."""
import os, numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.environ["MATSUM_OUT"]; GEO = os.environ["GEO_OUT"]
MIN_VISITS, MIN_SESS = 10, 3

S = pd.read_parquet(os.path.join(GEO, "geolife_sessions.parquet"))   # all 18670 sessions
P = pd.read_parquet(os.path.join(GEO, "geolife_points.parquet"))     # kept (>=80%), clipped
gdf = gpd.read_parquet(os.path.join(OUT, "trajectories.parquet"))
tiles = gpd.read_parquet(os.path.join(OUT, "enriched_tiles.parquet")).reset_index()

# semantic mapping (light) -> per-point tile; visits/session via RLE of category
sem = gpd.sjoin(gdf[["tid", "time", "geometry"]], tiles[["category", "label_tfidf", "geometry"]],
                predicate="within")
sem = sem[~sem.index.duplicated(keep="first")].sort_values(["tid", "time"])
coverage = len(sem) / len(gdf)
sem["user"] = sem["tid"].str.split("__").str[0]
sem["top1"] = sem["label_tfidf"].apply(lambda x: x[0] if hasattr(x, "__len__") and len(x) else "NA")
def nvis(a):
    a = a.to_numpy(); return int(1 + (a[1:] != a[:-1]).sum()) if len(a) else 0
vis = sem.groupby("tid")["category"].agg(nvis)
su = sem.groupby("tid")["user"].first()
vd = pd.DataFrame({"visits": vis, "user": su})
valid = vd[vd.visits >= MIN_VISITS]                                   # sessions with >=10 sem visits
valid_per_user = valid.groupby("user").size()
final_users = set(valid_per_user[valid_per_user >= MIN_SESS].index)
final_sessions = valid[valid.user.isin(final_users)]

# points per kept session (in-box, clipped)
pts_per_sess = P.groupby("tid").size()
inbox_pts = (S.npoints * S.frac_inbox)

def row(users, sessions, points):
    return {"users": users, "sessions": sessions, "points": int(points)}

ret = pd.DataFrame([
    row(182, len(S), S.npoints.sum()),                                                   # original
    row(S[S.frac_inbox > 0].user.nunique(), int((S.frac_inbox > 0).sum()), inbox_pts[S.frac_inbox > 0].sum()),
    row(P.user.nunique(), P.tid.nunique(), len(P)),                                       # >=80% in bbox (clipped)
    row(valid.user.nunique(), len(valid), pts_per_sess.reindex(valid.index).fillna(0).sum()),   # >=10 sem visits
    row(len(final_users), len(final_sessions), pts_per_sess.reindex(final_sessions.index).fillna(0).sum()),  # >=3 sessions/user
], index=["GeoLife original", ">=1 point in bbox", ">=80% in bbox",
          ">=10 semantic visits", ">=3 sessions/user (final)"])
ret.to_csv(os.path.join(OUT, "geolife_retention.csv"))

print("\n================  Retention table  ================")
print(ret.to_string())
per_user_visits = vd.groupby("user")["visits"].sum()
print("\n--- semantic coverage ---")
print(f"points with >=1 OSM label     : {coverage*100:.1f}%")
print(f"points without any label      : {(1-coverage)*100:.1f}%")
print(f"distinct OSM labels           : {sem['top1'].nunique()}")
print(f"median sessions/user (final)  : {valid_per_user[valid_per_user>=MIN_SESS].median():.0f}")
print(f"median sem visits/user (final): {per_user_visits[list(final_users)].median():.0f}")

# ---- 3 diagnostic plots ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
vpu = valid_per_user[valid_per_user >= MIN_SESS]
ax[0].hist(vpu, bins=np.logspace(np.log10(3), np.log10(vpu.max()), 25), color="#2c7fb8")
ax[0].set_xscale("log"); ax[0].set_title("(A) valid sessions per user"); ax[0].set_xlabel("sessions (log)"); ax[0].set_ylabel("users")
ax[1].hist(valid.visits, bins=np.logspace(1, np.log10(valid.visits.max()), 30), color="#2f7d4f")
ax[1].set_xscale("log"); ax[1].set_title("(B) semantic visits per session"); ax[1].set_xlabel("visits (log)"); ax[1].set_ylabel("sessions")
samp = P.sample(min(300000, len(P)), random_state=0)
hb = ax[2].hexbin(samp.lon, samp.lat, gridsize=60, bins="log", cmap="viridis")
ax[2].set_title("(C) coverage of kept sessions"); ax[2].set_xlabel("lon"); ax[2].set_ylabel("lat"); ax[2].set_aspect("equal")
fig.colorbar(hb, ax=ax[2], shrink=.8, label="log point density")
fig.suptitle("GeoLife study-area audit (frozen filters)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(os.environ.get("MATSUM_FIG", OUT), "geolife_diagnostics.png")
os.makedirs(os.path.dirname(p), exist_ok=True); fig.savefig(p, dpi=130); plt.close(fig)
print("\nfigure:", p)
