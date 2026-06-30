"""Aggregate tile_detect.py JSON records into a cross-device comparison table.

Loads every ``*.json`` under ``--data-dir`` (default ``tile_detect_data``,
where tile_detect.py writes them) and prints one row per run: the GEMM tile,
the op's ripple period / knee / argmin, the GEMM<->op ripple correlation, and
whether the op tracked the GEMM tile. Sorted by device then shape so records
collected from different machines line up.

The headline questions this answers as data accumulates:
  * does the op track the GEMM tile on every GPU (r high, period == T)?
  * how does the tile T vary with compute capability / SM count?
  * is the tile shape- or dtype-dependent on a given device?

    python tile_detect_collect.py --data-dir tile_detect_data
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="tile_detect_data")
    p.add_argument("--rederive", action="store_true",
                   help="recompute derived metrics from each record's raw sweep "
                        "(via tile_detect.analyze) and rewrite it in place")
    p.add_argument("--plot", action="store_true",
                   help="render the 3-panel figure for each record from its raw "
                        "sweep into --data-dir (regenerable without hardware)")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    recs = []
    for path in paths:
        with open(path) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"no records in {args.data_dir}")
        return

    if args.rederive:
        from tile_detect import analyze
        for path, r in zip(paths, recs):
            r.update(analyze(r["Ms"], r["gemm_nspr"], r["op_nspr"],
                             r.get("m_step", 16), r.get("m_max", max(r["Ms"]))))
            r["schema"] = 3
            with open(path, "w") as f:
                json.dump(r, f, indent=2)
        print(f"re-derived {len(recs)} record(s) from raw\n")

    if args.plot:
        from tile_detect import plot_record
        for r in recs:
            plot_record(r, args.data_dir)
        print()

    def cc(r):
        c = r.get("compute_capability")
        return f"{c[0]}.{c[1]}" if c else "-"

    def fmt(v):
        return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    recs.sort(key=lambda r: (r["device_name"], r["dtype"], r["N"], r["K"]))
    hdr = ["device", "cc", "sm", "dtype", "K", "N", "NT",
           "GEMM_T", "reg", "op_period", "r", "pen%", "tracks"]
    w = [23, 5, 4, 9, 7, 8, 7, 8, 5, 11, 7, 7, 8]
    print("".join(h.rjust(c) for h, c in zip(hdr, w)))
    for r in recs:
        row = [r["device_name"][:22], cc(r), str(r.get("sm_count", "-")),
               r["dtype"], r["K"], r["N"], r["NT"],
               fmt(r["gemm_tile_T"]),
               "yes" if r.get("gemm_tile_regular") else "no",
               fmt(r["op_ripple_period"]), fmt(r["ripple_corr_r"]),
               fmt(r.get("ripple_penalty_pct")),
               "yes" if r["op_tracks_gemm_tile"] else "no"]
        print("".join(str(v).rjust(c) for v, c in zip(row, w)))

    gpu = [r for r in recs if r["device_type"] == "cuda"]
    if gpu:
        tracks = sum(r["op_tracks_gemm_tile"] for r in gpu)
        rs = [r["ripple_corr_r"] for r in gpu if r["ripple_corr_r"] is not None]
        rrange = f"[{min(rs):.2f}, {max(rs):.2f}]" if rs else "n/a"
        print(f"\nGPU runs: {len(gpu)} | op-tracks-tile {tracks}/{len(gpu)} | "
              f"r in {rrange}")
        reg = [r for r in gpu if r.get("gemm_tile_regular")]
        if reg:
            print("  regular-sawtooth tiles (cc, shape -> tile, mis-align penalty):")
            for r in sorted(reg, key=lambda r: (cc(r), r["N"], r["K"])):
                print(f"    cc{cc(r):<4} N={r['N']:<7} K={r['K']:<5} -> tile "
                      f"{r['gemm_tile_T']:<4} ~{fmt(r.get('ripple_penalty_pct'))}% penalty")


if __name__ == "__main__":
    main()
