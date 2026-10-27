"""Summarize data/check_alignment.py output against the self-grounding verdicts (no GPU).

Reports (1) whether the instruction noun matches the dataset's own object list, (2) which mask orientation fits the
grasp centers, and (3) for objects kept vs flagged by the filter, where their grasps lie: in the object's own mask,
in another object's mask, or in no mask.

Usage: python analysis/alignment_summary.py data/gapp5k runs/align
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.data import object_name  # noqa: E402


def main():
    subset, align_dir = sys.argv[1:3]
    align = {(a["scene"], a["obj"]): a for a in map(json.loads, open(os.path.join(align_dir, "alignment.jsonl")))}
    verdicts = {(v["scene"], v["obj"]): v["verdict"]
                for v in map(json.loads, open(os.path.join(subset, "ground_verdicts.jsonl")))}

    same = sum(a["v1_name"] is not None and a["v1_name"].lower() == object_name(a["prompts"][0]) for a in align.values())
    print(f"instruction noun == dataset object list entry: {same} / {len(align)}")
    for tag in ("yx", "T"):
        own = [a[f"own_mask_frac_{tag}"] for a in align.values() if a[f"own_mask_frac_{tag}"] is not None]
        print(f"mean share of grasp centers in own mask, orientation {tag}: {statistics.mean(own):.3f}")

    for group in ("ok", "flagged"):
        c = collections.Counter()
        for k, a in align.items():
            if (verdicts.get(k) == "ok") != (group == "ok"):
                continue
            own = a["own_mask_frac_yx"]
            others = [f for o, f in a["mask_fracs_yx"].items() if o != k[1] and f is not None]
            if own is None:
                c["no own mask"] += 1
            elif own >= 0.5:
                c["own mask"] += 1
            elif others and max(others) >= 0.5:
                c["another object's mask"] += 1
            else:
                c["no mask"] += 1
        n = sum(c.values())
        print(f"{group:8s} n={n}: " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
