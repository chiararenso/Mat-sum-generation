"""Build the filtered GeoLife point dataset: kept sessions (>=80% in the FROZEN Beijing bbox),
points clipped to the bbox, minimal cleaning (temporal sort, drop exact dups & invalid coords).
No cross-session concatenation: tid = user__session. Saves a parquet for enrichment/experiments."""
import os, glob, numpy as np, pandas as pd

DATA = os.environ["GEOLIFE_DATA"]; GEO = os.environ.get("GEO_OUT", ".")
LATMIN, LATMAX, LONMIN, LONMAX = 39.75, 40.10, 116.00, 116.80   # FROZEN study area
KEEP_FRAC = 0.80

S = pd.read_parquet(os.path.join(GEO, "geolife_sessions.parquet"))
kept = S[S.frac_inbox >= KEEP_FRAC].copy()
print(f"kept sessions: {len(kept)} / {len(S)}")

parts = []
for i, (u, sess) in enumerate(zip(kept.user, kept.session)):
    f = os.path.join(DATA, u, "Trajectory", sess)
    try:
        df = pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1, 5, 6],
                         names=["lat", "lon", "date", "time"])
    except Exception:
        continue
    df = df[(df.lat >= LATMIN) & (df.lat <= LATMAX) & (df.lon >= LONMIN) & (df.lon <= LONMAX)]
    df = df[(df.lat.between(-90, 90)) & (df.lon.between(-180, 180))]
    if len(df) == 0:
        continue
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["lat", "lon", "time"])
    if len(df) == 0:
        continue
    df["user"] = u; df["session"] = sess; df["tid"] = f"{u}__{sess}"
    parts.append(df[["tid", "user", "session", "lat", "lon", "time"]])
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(kept)} sessions", flush=True)

P = pd.concat(parts, ignore_index=True)
P.to_parquet(os.path.join(GEO, "geolife_points.parquet"))
print(f"\npoints: {len(P):,}  sessions: {P.tid.nunique()}  users: {P.user.nunique()}")
print(f"bbox lat[{P.lat.min():.4f},{P.lat.max():.4f}] lon[{P.lon.min():.4f},{P.lon.max():.4f}]")
vis = P.groupby("user").session.nunique()
print(f"sessions/user (kept): median={vis.median():.0f} min={vis.min()} max={vis.max()}")
print(f"users with >=3 sessions: {(vis>=3).sum()}")
