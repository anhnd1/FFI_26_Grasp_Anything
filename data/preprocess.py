"""Preprocess a built subset into two takes for training/evaluation.

Input:  <subset>/annotations.jsonl  (from build_subset.py; left untouched)
Output: <subset>/raw.jsonl          every sample, normalized + training targets
        <subset>/clean.jsonl        same, minus samples with illogical labels (whole samples removed, never single boxes)
        <subset>/prep_report.json   counts per rule and split

Cleaning rules (thresholds chosen from the data with data/audit_labels.py):
  R1 degenerate  a sample containing any grasp with w < 4 px or h < 2 px
  R2 copied      every sample of an object whose mean grasp center is < 1 px from another object's in the same scene.
                 GA++ sometimes names objects the generator never rendered and copies a real object's grasps to them;
                 we cannot tell which name is real, so all objects involved are removed.

Targets (one grasp per sample is needed for text SFT; all GT grasps are kept for evaluation):
  medoid     grasp with minimal total endpoint distance to the others (consensus)
  top_score  grasp with the highest GA++ quality score (net-torque proxy from the dataset pipeline)

Row schema: id, scene, obj, part, image, prompt, split, grasps [[x, y, w, h, theta]], scores, target {medoid, top_score}

Usage: python data/preprocess.py data/gapp5k
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.geometry import medoid_index, norm_theta  # noqa: E402

MIN_W, MIN_H = 4.0, 2.0
COPY_PX = 1.0


def normalize(r):
    scene, obj, part = r["id"].rsplit("_", 2)
    grasps = [[x, y, w, h, norm_theta(t)] for _, x, y, w, h, t in r["grasps"]]
    scores = [g[0] for g in r["grasps"]]
    return {"id": r["id"], "scene": scene, "obj": obj, "part": part, "image": r["image"], "prompt": r["prompt"],
            "split": r["split"], "grasps": grasps, "scores": scores,
            "target": {"medoid": medoid_index(grasps), "top_score": max(range(len(scores)), key=scores.__getitem__)}}


def copied_objects(rows):
    """(scene, obj) keys whose mean grasp center is within COPY_PX of another object in the same scene."""
    centers = {}
    for r in rows:
        key = (r["scene"], r["obj"])
        if key not in centers:
            g = r["grasps"]
            centers[key] = (sum(v[0] for v in g) / len(g), sum(v[1] for v in g) / len(g))
    by_scene = defaultdict(list)
    for (scene, obj), c in centers.items():
        by_scene[scene].append((obj, c))
    bad = set()
    for scene, objs in by_scene.items():
        for i, (oa, ca) in enumerate(objs):
            for ob, cb in objs[i + 1:]:
                if math.dist(ca, cb) < COPY_PX:
                    bad |= {(scene, oa), (scene, ob)}
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    args = ap.parse_args()

    raw = [normalize(json.loads(line)) for line in open(os.path.join(args.subset_dir, "annotations.jsonl"), encoding="utf-8")]
    degenerate = {r["id"] for r in raw if any(g[2] < MIN_W or g[3] < MIN_H for g in r["grasps"])}
    copied = copied_objects(raw)

    clean, reasons = [], {}
    for r in raw:
        why = [n for n, hit in (("R1_degenerate", r["id"] in degenerate),
                                ("R2_copied", (r["scene"], r["obj"]) in copied)) if hit]
        if why:
            reasons[r["id"]] = why
        else:
            clean.append(r)

    for name, rows in (("raw", raw), ("clean", clean)):
        with open(os.path.join(args.subset_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def counts(rows):
        out = {}
        for split in ("train", "val", "test"):
            s = [r for r in rows if r["split"] == split]
            out[split] = {"samples": len(s), "objects": len({(r["scene"], r["obj"]) for r in s}),
                          "scenes": len({r["scene"] for r in s})}
        return out

    split_of = {r["id"]: r["split"] for r in raw}
    report = {
        "thresholds": {"min_w_px": MIN_W, "min_h_px": MIN_H, "copy_center_px": COPY_PX},
        "raw": counts(raw),
        "clean": counts(clean),
        "removed_by_rule": {rule: dict(Counter(split_of[i] for i, rs in reasons.items() if rule in rs))
                            for rule in ("R1_degenerate", "R2_copied")},
        "medoid_equals_top_score": round(sum(r["target"]["medoid"] == r["target"]["top_score"] for r in raw) / len(raw), 4),
    }
    with open(os.path.join(args.subset_dir, "prep_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
