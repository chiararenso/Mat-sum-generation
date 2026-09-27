"""Build geolife_sess_cache.pkl (sig2/top1 sequences per session, keyed by tid) from the
prepared GeoLife parquet files, using the exact same validated load() as geolife_e1/e3.py.
Run once after matsum_prepare.py + geolife_e0_semantic.py; downstream scripts (geolife_e1..e4,
geolife_disclosure.py, geolife_mia.py, geolife_release_size.py) reuse this cache."""
import os, pickle, numpy as np
from geolife_e1 import load

OUT = os.environ["MATSUM_OUT"]; GEO = os.environ["GEO_OUT"]


def main():
    sem = load()
    sess = {t: (g["sig2"].tolist(), g["top1"].tolist(), g["user"].iloc[0])
            for t, g in sem.groupby("tid", sort=False)}
    users = np.array(sorted(sem.user.unique()))
    cache = os.path.join(GEO, "geolife_sess_cache.pkl")
    with open(cache, "wb") as f:
        pickle.dump((sess, users), f)
    print(f"saved {cache}: {len(sess)} sessions, {len(users)} users")


if __name__ == "__main__":
    main()
