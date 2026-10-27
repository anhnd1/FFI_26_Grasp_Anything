"""Label-noise audit of a built subset (annotations.jsonl from build_subset.py).

Reports, per split and overall:
  - objects (scene, obj) and pairs (scene, obj, part)
  - share of objects whose part prompts all have identical grasp sets
  - share of multi-object scenes where two different objects have mean grasp centers < --px apart

Usage: python data/audit_labels.py data/gapp5k/annotations.jsonl [--px 10] [--json out.json]
"""
import argparse
import json
from collections import Counter, defaultdict


def grasp_sig(grasps):
    # Drop the score column; round to 1 px / 1 degree so float noise doesn't hide duplicates.
    return tuple(sorted(tuple(round(v) for v in g[1:]) for g in grasps))


def center(grasps):
    return sum(g[1] for g in grasps) / len(grasps), sum(g[2] for g in grasps) / len(grasps)


def audit(rows, px):
    scenes = defaultdict(lambda: defaultdict(list))
    for r in rows:
        scene, obj, _ = r["id"].rsplit("_", 2)
        scenes[scene][obj].append(r["grasps"])

    multi_part = identical = multi_obj = collide = 0
    for objs in scenes.values():
        for parts in objs.values():
            if len(parts) > 1:
                multi_part += 1
                identical += len({grasp_sig(g) for g in parts}) == 1
        if len(objs) > 1:
            multi_obj += 1
            cs = [center(parts[0]) for parts in objs.values()]
            d = min(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for i, a in enumerate(cs) for b in cs[i + 1:])
            collide += d < px

    return {
        "scenes": len(scenes),
        "objects": sum(len(o) for o in scenes.values()),
        "pairs": len(rows),
        "objects_per_scene": dict(sorted(Counter(len(o) for o in scenes.values()).items())),
        "objects_multi_part": multi_part,
        "objects_identical_parts": identical,
        "identical_parts_rate": round(identical / max(1, multi_part), 4),
        "scenes_multi_object": multi_obj,
        "scenes_center_collision": collide,
        "center_collision_rate": round(collide / max(1, multi_obj), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("--px", type=float, default=10.0)
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.annotations, encoding="utf-8")]
    report = {"all": audit(rows, args.px)}
    for split in ("train", "val", "test"):
        report[split] = audit([r for r in rows if r["split"] == split], args.px)
    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
