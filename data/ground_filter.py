"""Self-grounding label filter, step 2 (local, no GPU): flag objects whose GT grasps lie outside Qwen's boxes.

Reads ground_*.jsonl from ground.py. An object is kept if at least --min-frac of its GT grasp centers (all its part
prompts) fall inside one of the boxes Qwen drew for its name (boxes padded by --pad of their size + 4 px). Verdicts:
  ok       enough grasp centers inside a box
  outside  Qwen found the object, but the grasps are elsewhere (label on another object)
  nobox    Qwen returned no box for the name (object missing/distorted, or an unparseable answer)

Writes into the subset dir:
  ground_verdicts.jsonl  one line per object: scene, obj, split, noun, verdict, frac, boxes (pixels)
  ground.jsonl           the "ground" take: raw rows minus flagged objects minus R1 (degenerate-grasp) samples
  ground_report.json     counts per split and verdict
The reported "verified" test view removes the flagged test objects confirmed as mislabeled
(data/splits/verified_test.json, scored by analysis/score_subset.py).

Usage: python data/ground_filter.py data/gapp5k runs/ground [--coords auto|norm|pixel] [--viz 12]
"""
import argparse
import collections
import glob
import io
import json
import math
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.geometry import corners  # noqa: E402
from preprocess import MIN_H, MIN_W  # noqa: E402

IMG = 416


def degenerate(r):
    """R1 of preprocess.py: a sample with any grasp narrower than MIN_W or thinner than MIN_H."""
    return any(g[2] < MIN_W or g[3] < MIN_H for g in r["grasps"])


def inside(pt, box, pad):
    x1, y1, x2, y2 = box
    px, py = pad * (x2 - x1) + 4, pad * (y2 - y1) + 4
    return x1 - px <= pt[0] <= x2 + px and y1 - py <= pt[1] <= y2 + py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    ap.add_argument("ground_dir")
    ap.add_argument("--coords", choices=["auto", "norm", "pixel"], default="auto")
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--viz", type=int, default=0, help="also draw N random flagged objects to ground_viz_<verdict>.png")
    args = ap.parse_args()

    preds = [json.loads(line) for p in sorted(glob.glob(os.path.join(args.ground_dir, "**", "ground_*.jsonl"),
                                                        recursive=True)) for line in open(p)]
    coords = args.coords
    if coords == "auto":  # Qwen3-style 0-1000 if any coordinate exceeds the 416 px image
        coords = "norm" if any(v > IMG + 4 for p in preds for b in p["boxes"] for v in b) else "pixel"
    scale = IMG / 1000 if coords == "norm" else 1.0
    print("coords:", coords, "objects:", len(preds))

    rows = [json.loads(line) for line in open(os.path.join(args.subset_dir, "raw.jsonl"), encoding="utf-8")]
    by_obj = collections.defaultdict(list)
    for r in rows:
        by_obj[(r["scene"], r["obj"])].append(r)

    verdicts = {}
    for p in preds:
        boxes = [[v * scale for v in b] for b in p["boxes"]]
        centers = [g[:2] for r in by_obj[(p["scene"], p["obj"])] for g in r["grasps"]]
        frac = sum(any(inside(c, b, args.pad) for b in boxes) for c in centers) / max(1, len(centers))
        v = "nobox" if not boxes else ("ok" if frac >= args.min_frac else "outside")
        verdicts[(p["scene"], p["obj"])] = {"scene": p["scene"], "obj": p["obj"], "split": p["split"], "noun": p["noun"],
                                            "verdict": v, "frac": round(frac, 3), "boxes": boxes, "text": p["text"]}

    with open(os.path.join(args.subset_dir, "ground_verdicts.jsonl"), "w") as f:
        for v in verdicts.values():
            f.write(json.dumps(v) + "\n")
    kept = [r for r in rows if verdicts.get((r["scene"], r["obj"]), {}).get("verdict") == "ok" and not degenerate(r)]
    with open(os.path.join(args.subset_dir, "ground.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    report = {"coords": coords, "min_frac": args.min_frac, "pad": args.pad, "objects": {}, "samples_kept": {}}
    for split in ("train", "val", "test"):
        c = collections.Counter(v["verdict"] for v in verdicts.values() if v["split"] == split)
        report["objects"][split] = dict(c, total=sum(c.values()))
        report["samples_kept"][split] = {"ground": sum(r["split"] == split for r in kept),
                                         "raw": sum(r["split"] == split for r in rows)}
    json.dump(report, open(os.path.join(args.subset_dir, "ground_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))

    if args.viz:
        from PIL import Image, ImageDraw
        z = zipfile.ZipFile(os.path.join(args.subset_dir, "images.zip"))
        for verdict in ("outside", "nobox", "ok"):
            items = [v for v in verdicts.values() if v["verdict"] == verdict]
            items = random.Random(0).sample(items, min(args.viz, len(items)))
            if not items:
                continue
            cols = 4
            sheet = Image.new("RGB", (IMG * cols, (IMG + 20) * math.ceil(len(items) / cols)), "black")
            for i, v in enumerate(items):
                r0 = by_obj[(v["scene"], v["obj"])][0]
                img = Image.open(io.BytesIO(z.read(r0["image"]))).convert("RGB")
                d = ImageDraw.Draw(img)
                for r in by_obj[(v["scene"], v["obj"])]:
                    for g in r["grasps"]:
                        d.polygon(corners(*g), outline=(0, 220, 0))
                for b in v["boxes"]:
                    d.rectangle(b, outline=(0, 160, 255), width=3)
                tile = Image.new("RGB", (IMG, IMG + 20), "black")
                tile.paste(img, (0, 0))
                ImageDraw.Draw(tile).text((4, IMG + 4), f"{v['noun']}  frac {v['frac']}  ({v['split']})", fill="white")
                sheet.paste(tile, ((i % cols) * IMG, (i // cols) * (IMG + 20)))
            out = os.path.join(args.ground_dir, f"ground_viz_{verdict}.png")
            sheet.save(out)
            print(out)


if __name__ == "__main__":
    main()
