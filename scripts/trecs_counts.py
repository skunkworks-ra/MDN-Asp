"""T-RECS medi counts per 256 px patch (2 arcsec cells), AGN and SFG, same cut as
sim_trecs_points_2048_fullfield_1e-5: I1400 >= 1e-5 Jy, |x|,|y| <= 0.5 deg.
Byte columns from radiosharp/data/ReadMe. pixi run python <this>"""
import argparse, gzip, os, time
from pathlib import Path
import numpy as np

def main(argv=None):
    p = argparse.ArgumentParser(description="Build trecs_counts.npz from T-RECS medi catalogues.")
    p.add_argument("--data_dir", default=os.environ.get("TRECS_DATA_DIR", "data/trecs"),
                   help="Directory holding agnsmedi.dat.gz and sfgsmedi.dat.gz (default: $TRECS_DATA_DIR or data/trecs).")
    p.add_argument("--out", default="trecs_counts.npz", help="Output .npz path.")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    FLOOR_MJY, HALF_DEG, CELL, PATCH = 1e-2, 0.5, 2.0, 256
    COLS = {"agn": dict(f=(113, 124), x=(467, 476), y=(478, 487), size=(554, 571)),
            "sfg": dict(f=(99, 108), x=(380, 389), y=(391, 400), size=(433, 450))}
    sl = lambda c: slice(c[0] - 1, c[1])
    patches_per_field = (2 * HALF_DEG * 3600.0 / (PATCH * CELL)) ** 2
    out = {}
    for pop, fn in (("agn", "agnsmedi.dat.gz"), ("sfg", "sfgsmedi.dat.gz")):
        c = COLS[pop]; F, S = [], []; t0 = time.time(); n = 0
        with gzip.open(data_dir / fn, "rt") as fh:
            for line in fh:
                n += 1
                if line.startswith("#") or len(line) < c["size"][1]:
                    continue
                try:
                    f = float(line[sl(c["f"])])
                    if f < FLOOR_MJY:
                        continue
                    x, y = float(line[sl(c["x"])]), float(line[sl(c["y"])])
                    if abs(x) > HALF_DEG or abs(y) > HALF_DEG:
                        continue
                    F.append(f); S.append(float(line[sl(c["size"])]))
                except ValueError:
                    continue
                if n % 5_000_000 == 0:
                    print(f"  {pop} {n:,} lines {time.time()-t0:.0f}s kept {len(F)}", flush=True)
        F, S = np.array(F) * 1e-3, np.array(S)
        out[pop] = (F, S)
        print(f"{pop}: {n:,} lines read in {time.time()-t0:.0f}s; in field above 1e-5 Jy: {len(F)}; "
              f"per {PATCH} px patch: {len(F)/patches_per_field:.2f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, agn_flux=out["agn"][0], agn_size=out["agn"][1],
                        sfg_flux=out["sfg"][0], sfg_size=out["sfg"][1])
    print(f"patches per field (1 deg^2 / ({PATCH} x {CELL} arcsec)^2): {patches_per_field:.3f}")
    edges = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e2]
    for pop, (F, S) in out.items():
        h, _ = np.histogram(F, edges)
        print(f"{pop} counts per flux decade (Jy) {edges}: {h.tolist()}  per patch: "
              f"{[round(v / patches_per_field, 2) for v in h]}")
        p_ = np.percentile(S / CELL, [10, 50, 90, 99]) if len(S) else []
        print(f"{pop} Size column in px (2 arcsec cells) p10/p50/p90/p99: {np.round(p_, 2).tolist()}  "
              f"fraction > 1 px: {np.mean(S / CELL > 1):.3f}")


if __name__ == "__main__":
    main()
