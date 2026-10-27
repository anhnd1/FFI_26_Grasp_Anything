"""Constant-grasp baseline: predict the same rectangle for every test sample (no model, no GPU).

The rectangle sits at the image center with the median size and the modal angle (22.5 deg bins) of the training
targets (the medoid grasp of each train sample),
which calibrates how much of the metric a model gets for free (GA++ objects tend to be centered).

Usage: python analysis/constant_baseline.py data/gapp5k [data/splits/verified_test.json]
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.data import load_rows  # noqa: E402
from grasp.geometry import is_success  # noqa: E402

BIN = 22.5


def main():
    subset = sys.argv[1]
    train = [r["grasps"][r["target"]["medoid"]] for r in load_rows(subset, "raw", "train")]
    w = round(statistics.median(g[2] for g in train))
    h = round(statistics.median(g[3] for g in train))
    angle = collections.Counter(int(g[4] // BIN) for g in train).most_common(1)[0][0] * BIN
    grasp = [208, 208, w, h, angle]

    test = load_rows(subset, "raw", "test")
    views = {"raw": test, "clean": load_rows(subset, "clean", "test")}
    if len(sys.argv) > 2:
        removed = set(json.load(open(sys.argv[2]))["removed_objects"])
        views["verified"] = [r for r in test if f"{r['scene']}_{r['obj']}" not in removed]
    print("constant grasp [x, y, w, h, theta]:", grasp)
    for name, rows in views.items():
        print(f"{name:9s} {100 * sum(is_success(grasp, r['grasps']) for r in rows) / len(rows):.1f}%  (n={len(rows)})")


if __name__ == "__main__":
    main()
