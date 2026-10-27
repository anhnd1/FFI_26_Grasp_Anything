"""Score saved predictions on the raw, clean and verified test views (no GPU).

eval.py predicts once on the full raw test split; this script re-reads predictions.jsonl and reports success (%) on
  raw       all test samples
  clean     samples kept by the rule-based cleaning (data/preprocess.py)
  verified  raw test minus every sample of the objects listed in data/splits/verified_test.json
  removed   the complementary samples: "success" there means the model reproduced a label that is on the wrong object
plus the number of degenerate predictions (w < 4 px or h < 2 px).

Usage: python analysis/score_subset.py data/gapp5k data/splits/verified_test.json name=runs/ep_ground/test ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.data import load_rows  # noqa: E402

MIN_W, MIN_H = 4.0, 2.0


def main():
    subset, verified_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    removed = set(json.load(open(verified_path))["removed_objects"])
    rows = load_rows(subset, "raw", "test")
    is_removed = {r["id"]: f"{r['scene']}_{r['obj']}" in removed for r in rows}
    print(f"test samples: raw {len(rows)}, verified {sum(not v for v in is_removed.values())}, "
          f"removed {sum(is_removed.values())} ({len(removed)} objects)")

    print(f"{'run':32s} {'raw':>6s} {'clean':>6s} {'verif':>6s} {'removed':>8s} {'degenerate':>11s}")
    for spec in specs:
        name, path = spec.split("=", 1)
        preds = [json.loads(line) for line in open(os.path.join(path, "predictions.jsonl"))]

        def rate(ps):
            return 100 * sum(p["success"] for p in ps) / max(1, len(ps))
        degenerate = sum(1 for p in preds if p["pred"] and (p["pred"][2] < MIN_W or p["pred"][3] < MIN_H))
        print(f"{name:32s} {rate(preds):6.1f} {rate([p for p in preds if p['clean']]):6.1f} "
              f"{rate([p for p in preds if not is_removed[p['id']]]):6.1f} "
              f"{rate([p for p in preds if is_removed[p['id']]]):8.1f} {degenerate:11d}")


if __name__ == "__main__":
    main()
