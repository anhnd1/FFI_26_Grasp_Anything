"""Failure analysis for eval.py predictions (local, no GPU).

For each run: success split by failure mode (fails only the angle test / only the IoU test / both), the share of
samples where the prediction is wrong for the prompted object but would count as a success for ANOTHER object in the
scene (wrong_object = grounding error), success for samples whose
ground-truth medoid endpoints fall outside the image vs inside, center error, and pairwise agreement
between runs (how often one is right where the other is wrong).

Usage: python analysis/analyze.py data/gapp5k ours=runs/ep_ground/test raw=runs/ep_raw/test [--json out.json]
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.data import load_rows  # noqa: E402
from grasp.geometry import angle_diff, iou, is_success, to_endpoints, to_norm  # noqa: E402


def out_of_frame(g):
    p1, p2, _ = to_endpoints(*g)
    return any(not 0 <= to_norm(v) <= 1000 for v in (*p1, *p2))


def mcnemar_p(b, c):
    """Paired test on discordant counts b, c (chi-square, 1 dof, continuity-corrected)."""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return math.erfc(math.sqrt(chi2 / 2))


def analyze(preds, rows, others):
    n = len(preds)
    modes = {"success": 0, "fail_angle_only": 0, "fail_iou_only": 0, "fail_both": 0, "parse": 0}
    wrong_object = 0
    oof = {True: [0, 0], False: [0, 0]}
    center_err = []
    for p in preds:
        r = rows[p["id"]]
        o = out_of_frame(r["grasps"][r["target"]["medoid"]])
        oof[o][0] += 1
        oof[o][1] += p["success"]
        if p["pred"] is None:
            modes["parse"] += 1
            continue
        g = p["pred"]
        center_err.append(min(math.dist(g[:2], t[:2]) for t in r["grasps"]))
        if p["success"]:
            modes["success"] += 1
            continue
        # best case over GTs for each criterion separately
        ang_ok = any(angle_diff(g[4], t[4]) < 30 for t in r["grasps"])
        iou_ok = any(iou(g, t) > 0.25 for t in r["grasps"])
        modes["fail_iou_only" if ang_ok else ("fail_angle_only" if iou_ok else "fail_both")] += 1
        wrong_object += any(is_success(g, gts) for gts in others[(r["scene"], r["obj"])])
    center_err.sort()
    return {
        "n": n,
        "modes": {k: round(v / n, 4) for k, v in modes.items()},
        "wrong_object": round(wrong_object / n, 4),
        "success_gt_endpoints_outside": round(oof[True][1] / max(1, oof[True][0]), 4),
        "success_gt_endpoints_inside": round(oof[False][1] / max(1, oof[False][0]), 4),
        "n_outside": oof[True][0],
        "center_err_px_median": round(center_err[len(center_err) // 2], 1) if center_err else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    ap.add_argument("runs", nargs="+", help="name=path/to/eval_dir")
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = {r["id"]: r for r in load_rows(args.subset_dir, "raw", "test")}
    obj_grasps = {}
    for r in rows.values():
        obj_grasps.setdefault((r["scene"], r["obj"]), r["grasps"])
    # grasp sets of the OTHER objects in the same scene, per (scene, obj)
    others = {k: [g for (s, o), g in obj_grasps.items() if s == k[0] and o != k[1]] for k in obj_grasps}
    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = [json.loads(line) for line in open(os.path.join(path, "predictions.jsonl"))]

    report = {}
    for name, preds in runs.items():
        for view in ("raw", "clean"):
            ps = preds if view == "raw" else [p for p in preds if p["clean"]]
            report[f"{name}/{view}"] = analyze(ps, rows, others)

    names = list(runs)
    pair = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = {p["id"]: p["success"] for p in runs[a]}
            sb = {p["id"]: p["success"] for p in runs[b]}
            ids = sa.keys() & sb.keys()
            only_a = sum(sa[i] and not sb[i] for i in ids)
            only_b = sum(sb[i] and not sa[i] for i in ids)
            pair[f"{a} vs {b}"] = {"both": sum(sa[i] and sb[i] for i in ids), "only_" + a: only_a, "only_" + b: only_b,
                                   "neither": sum(not sa[i] and not sb[i] for i in ids),
                                   "mcnemar_p": round(mcnemar_p(only_a, only_b), 5)}
    report["pairwise_raw"] = pair
    print(json.dumps(report, indent=2))
    if args.json:
        json.dump(report, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
