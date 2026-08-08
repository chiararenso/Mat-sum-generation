"""GeoLife E0 raw audit: parse all .plt, count users/sessions/points, sessions-per-user,
spatial percentiles (to freeze the Beijing study bbox), and in-area fraction per session.
NO cross-session concatenation. Saves per-session summary + a downsampled point cloud.
"""
import os, glob, numpy as np, pandas as pd

DATA = os.environ["GEOLIFE_DATA"]         # .../Geolife Trajectories 1.3/Data
OUT = os.environ.get("GEO_OUT", ".")
# candidate frozen study bbox (central Beijing) -- validated against percentiles below
LATMIN, LATMAX, LONMIN, LONMAX = 39.75, 40.10, 116.00, 116.80

recs = []; cloud_lat = []; cloud_lon = []
users = sorted(os.listdir(DATA))
for ui, u in enumerate(users):
    tdir = os.path.join(DATA, u, "Trajectory")
    if not os.path.isdir(tdir):
        continue
    for f in glob.glob(os.path.join(tdir, "*.plt")):
        try:
            df = pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1, 5],
                             names=["lat", "lon", "date"])
        except Exception:
            continue
        if len(df) == 0:
            continue
        inbox = ((df.lat >= LATMIN) & (df.lat <= LATMAX) &
                 (df.lon >= LONMIN) & (df.lon <= LONMAX)).mean()
        recs.append({"user": u, "session": os.path.basename(f), "npoints": len(df),
                     "frac_inbox": float(inbox), "start": str(df.date.iloc[0]),
                     "mlat": float(df.lat.mean()), "mlon": float(df.lon.mean())})
        cloud_lat.append(df.lat.values[::100]); cloud_lon.append(df.lon.values[::100])
    if (ui + 1) % 30 == 0:
        print(f"  parsed {ui+1}/{len(users)} users", flush=True)

S = pd.DataFrame(recs); S.to_parquet(os.path.join(OUT, "geolife_sessions.parquet"))
lat = np.concatenate(cloud_lat); lon = np.concatenate(cloud_lon)
pd.DataFrame({"lat": lat, "lon": lon}).to_parquet(os.path.join(OUT, "geolife_cloud.parquet"))

print("\n===== GeoLife raw audit =====")
print(f"original users            : {len(users)}")
print(f"total sessions (.plt)     : {len(S)}")
print(f"total points              : {int(S.npoints.sum()):,}")
spu = S.groupby('user').size()
print(f"sessions/user  median={spu.median():.0f}  mean={spu.mean():.1f}  min={spu.min()} max={spu.max()}")
print("\nspatial percentiles (downsampled cloud) -- to freeze the bbox:")
for q in [0.5, 1, 2.5, 50, 97.5, 99, 99.5]:
    print(f"  p{q:>4}: lat={np.percentile(lat,q):.4f}  lon={np.percentile(lon,q):.4f}")
print(f"\ncandidate bbox lat[{LATMIN},{LATMAX}] lon[{LONMIN},{LONMAX}]")
frac_cloud_in = ((lat>=LATMIN)&(lat<=LATMAX)&(lon>=LONMIN)&(lon<=LONMAX)).mean()
print(f"  cloud points inside bbox : {frac_cloud_in*100:.1f}%")
kept = S[S.frac_inbox >= 0.80]
print(f"  sessions with >=80% in bbox: {len(kept)} / {len(S)} ({len(kept)/len(S)*100:.1f}%)")
print(f"  points in kept sessions   : {int(kept.npoints.sum()):,}")
uk = kept.groupby('user').size()
print(f"  users with >=1 kept session : {uk.size}")
print(f"  users with >=3 kept sessions: {(uk>=3).sum()}   <-- candidate U")
