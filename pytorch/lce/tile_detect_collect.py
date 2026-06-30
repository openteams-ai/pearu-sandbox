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
    args = p.parse_args()

    recs = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.json"))):
        with open(path) as f:
            recs.append(json.load(f))
    if not recs:
        print(f"no records in {args.data_dir}")
        return

    def cc(r):
        c = r.get("compute_capability")
        return f"{c[0]}.{c[1]}" if c else "-"

    recs.sort(key=lambda r: (r["device_name"], r["dtype"], r["N"], r["K"]))
    hdr = ["device", "cc", "sm", "dtype", "K", "N", "NT",
           "GEMM_T", "op_period", "op_knee", "argmin", "r", "tracks"]
    w = [23, 5, 4, 9, 7, 8, 7, 8, 11, 9, 8, 7, 8]
    print("".join(h.rjust(c) for h, c in zip(hdr, w)))
    for r in recs:
        row = [r["device_name"][:22], cc(r), str(r.get("sm_count", "-")),
               r["dtype"], r["K"], r["N"], r["NT"],
               r["gemm_tile_T"], r["op_ripple_period"], r["op_knee"],
               r["op_argmin"], f"{r['ripple_corr_r']:.2f}",
               "yes" if r["op_tracks_gemm_tile"] else "no"]
        print("".join(str(v).rjust(c) for v, c in zip(row, w)))

    gpu = [r for r in recs if r["device_type"] == "cuda"]
    if gpu:
        tracks = sum(r["op_tracks_gemm_tile"] for r in gpu)
        match = sum(r["gemm_tile_T"] == r["op_ripple_period"] for r in gpu)
        rs = [r["ripple_corr_r"] for r in gpu]
        print(f"\nGPU runs: {len(gpu)} | op-tracks-tile {tracks}/{len(gpu)} | "
              f"period==T {match}/{len(gpu)} | r in [{min(rs):.2f}, {max(rs):.2f}]")
        by_cc = {}
        for r in gpu:
            by_cc.setdefault(cc(r), set()).add(r["gemm_tile_T"])
        print("  GEMM tile by compute capability: "
              + ", ".join(f"cc{k}->{sorted(v)}" for k, v in sorted(by_cc.items())))


if __name__ == "__main__":
    main()
