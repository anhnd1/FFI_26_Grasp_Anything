"""Draw GT grasp rectangles on subset images, one color per object, prompts as captions.

Usage: python data/visualize.py data/gapp5k [--n 6] [--multi-object | --flagged 10] [--seed 0] [--out viz.png]

--flagged PX shows only scenes where two different objects have mean grasp centers < PX apart (the
"center collision" heuristic in audit_labels.py); tiles are numbered for manual review.
"""
import argparse
import collections
import io
import json
import math
import os
import random
import zipfile

from PIL import Image, ImageDraw

from audit_labels import center

COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange"]


def corners(x, y, w, h, theta_deg):
    """Rectangle corners; w runs along theta (jaw opening), h across it."""
    a = math.radians(theta_deg)
    dx, dy = math.cos(a), math.sin(a)
    return [(x + sw * w / 2 * dx - sh * h / 2 * dy, y + sw * w / 2 * dy + sh * h / 2 * dx)
            for sw, sh in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def draw_scene(img, rows):
    d = ImageDraw.Draw(img)
    objs = sorted({r["id"].rsplit("_", 2)[1] for r in rows})
    captions = []
    for r in rows:
        color = COLORS[objs.index(r["id"].rsplit("_", 2)[1]) % len(COLORS)]
        if r["id"].endswith("_0"):
            captions.append((color, r["prompt"]))
        for _, x, y, w, h, t in r["grasps"]:
            d.polygon(corners(x, y, w, h, t), outline=color)
    for i, (color, text) in enumerate(captions):
        d.text((4, 4 + 12 * i), text, fill=color)
    return img


def min_center_dist(rows):
    """Smallest distance between mean grasp centers of two different objects (inf if one object)."""
    per_obj = {}
    for r in rows:
        per_obj.setdefault(r["id"].rsplit("_", 2)[1], r["grasps"])
    cs = [center(g) for g in per_obj.values()]
    return min((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for i, a in enumerate(cs) for b in cs[i + 1:]),
               default=float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--multi-object", action="store_true", help="only scenes with >= 2 objects")
    ap.add_argument("--flagged", type=float, help="only scenes with a cross-object center collision < PX")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    by_scene = collections.defaultdict(list)
    for line in open(os.path.join(args.subset_dir, "annotations.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        by_scene[r["scene"]].append(r)
    scenes = [s for s, rows in by_scene.items()
              if not args.multi_object or len({r["id"].rsplit("_", 2)[1] for r in rows}) > 1]
    if args.flagged is not None:
        scenes = [s for s in scenes if min_center_dist(by_scene[s]) < args.flagged]
    scenes = random.Random(args.seed).sample(sorted(scenes), args.n)

    z = zipfile.ZipFile(os.path.join(args.subset_dir, "images.zip"))
    cols = 3
    sheet = Image.new("RGB", (416 * cols, 416 * math.ceil(args.n / cols)))
    for i, s in enumerate(scenes):
        img = draw_scene(Image.open(io.BytesIO(z.read(by_scene[s][0]["image"]))).convert("RGB"), by_scene[s])
        if args.flagged is not None:
            ImageDraw.Draw(img).text((4, 400), f"#{i + 1}  d={min_center_dist(by_scene[s]):.1f}px", fill="white")
        sheet.paste(img, ((i % cols) * 416, (i // cols) * 416))
    out = args.out or os.path.join(args.subset_dir, "viz.png")
    sheet.save(out)
    print(out, scenes)


if __name__ == "__main__":
    main()
