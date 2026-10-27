"""Visualize predictions of one eval run by outcome, and split IoU failures into position vs size errors (no GPU).

Tiles: green = every GT grasp of the prompted object, gray = GT grasps of other objects in the scene,
yellow = the GT grasp that best matches the prediction (max IoU among angle-compatible GTs), red = prediction.
Caption: prompt, best IoU, center error to that GT, predicted vs GT size (w x h, px).

Categories: iou   = failed only because IoU <= 0.25 (angle within 30 deg of some GT), right object
            wrong = prediction would be a success for another object in the scene
            ok    = success

Also prints, for IoU-only failures, the IoU after moving the prediction onto the matched GT's center
("position fixed") and after copying the GT's w, h instead ("size fixed"), to show which error dominates, and the
center offset split into the component along the jaw opening (w axis) and along the jaw plates (h axis).

Usage: python analysis/viz_failures.py data/gapp5k runs/ep_raw/test [--n 9] [--seed 0] [--out-dir DIR]
"""
import argparse
import collections
import io
import json
import math
import os
import random
import sys
import zipfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.data import load_rows  # noqa: E402
from grasp.geometry import angle_diff, corners, iou, is_success  # noqa: E402

CAP = 46  # caption height under each 416 px tile


def matched_gt(g, gts):
    """GT with max IoU among those within 30 deg (falls back to all GTs)."""
    cands = [t for t in gts if angle_diff(g[4], t[4]) < 30] or gts
    return max(cands, key=lambda t: iou(g, t))


def category(p, r, others):
    if p["pred"] is None:
        return "parse"
    if p["success"]:
        return "ok"
    if any(is_success(p["pred"], gts) for gts in others):
        return "wrong"
    if any(angle_diff(p["pred"][4], t[4]) < 30 for t in r["grasps"]):
        return "iou"
    return "other"


def tile(z, p, r, others):
    img = Image.open(io.BytesIO(z.read(r["image"]))).convert("RGB")
    canvas = Image.new("RGB", (416, 416 + CAP), "black")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for gts in others:
        for t in gts:
            d.polygon(corners(*t), outline=(150, 150, 150))
    for t in r["grasps"]:
        d.polygon(corners(*t), outline=(0, 200, 0))
    g = p["pred"]
    m = matched_gt(g, r["grasps"])
    d.polygon(corners(*m), outline="yellow", width=3)
    d.polygon(corners(*g), outline="red", width=3)
    d.text((4, 420), r["prompt"][:70], fill="white")
    d.text((4, 434), f"IoU {iou(g, m):.2f}  ctr err {math.dist(g[:2], m[:2]):.0f}px  "
                     f"pred {g[2]:.0f}x{g[3]:.0f}  gt {m[2]:.0f}x{m[3]:.0f}  ({len(r['grasps'])} GT)", fill="white")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_dir")
    ap.add_argument("eval_dir")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    rows = {r["id"]: r for r in load_rows(args.subset_dir, "raw", "test")}
    obj_grasps = {}
    for r in rows.values():
        obj_grasps.setdefault((r["scene"], r["obj"]), r["grasps"])
    preds = [json.loads(line) for line in open(os.path.join(args.eval_dir, "predictions.jsonl"))]

    groups = {}
    for p in preds:
        r = rows[p["id"]]
        others = [g for (s, o), g in obj_grasps.items() if s == r["scene"] and o != r["obj"]]
        groups.setdefault(category(p, r, others), []).append((p, r, others))
    print({k: len(v) for k, v in groups.items()})

    # position vs size decomposition on IoU-only failures
    fixed_pos = fixed_size = 0
    ratios, along, across = [], [], []
    bins = collections.Counter()  # center error to the matched GT
    for p, r, _ in groups.get("iou", []):
        g = p["pred"]
        m = matched_gt(g, r["grasps"])
        bins[next(b for b in (16, 32, 64, 128, math.inf) if math.dist(g[:2], m[:2]) < b)] += 1
        fixed_pos += iou([m[0], m[1], *g[2:]], m) > 0.25
        fixed_size += iou([*g[:2], m[2], m[3], g[4]], m) > 0.25
        ratios.append(g[2] / m[2])
        a = math.radians(m[4])
        dx, dy = g[0] - m[0], g[1] - m[1]
        along.append(abs(dx * math.cos(a) + dy * math.sin(a)))  # along the jaw opening (w axis)
        across.append(abs(-dx * math.sin(a) + dy * math.cos(a)))  # along the jaw plate (h axis)
    n = max(1, len(groups.get("iou", [])))
    for v in (ratios, along, across):
        v.sort()
    print(f"IoU-only failures: {n}; success if center moved onto matched GT: {fixed_pos / n:.1%}; "
          f"if size copied from GT: {fixed_size / n:.1%}; median pred/GT opening ratio {ratios[len(ratios) // 2]:.2f}; "
          f"median |offset| along w {along[len(along) // 2]:.0f}px, along h {across[len(across) // 2]:.0f}px")
    print("center error of IoU-only failures (< px):", {b: f"{bins[b] / n:.0%}" for b in sorted(bins)})

    z = zipfile.ZipFile(os.path.join(args.subset_dir, "images.zip"))
    out_dir = args.out_dir or args.eval_dir
    os.makedirs(out_dir, exist_ok=True)
    for cat in ("iou", "wrong", "ok"):
        items = groups.get(cat, [])
        items = random.Random(args.seed).sample(items, min(args.n, len(items)))
        cols = 3
        sheet = Image.new("RGB", (416 * cols, (416 + CAP) * math.ceil(len(items) / cols)))
        for i, (p, r, others) in enumerate(items):
            sheet.paste(tile(z, p, r, others), ((i % cols) * 416, (i // cols) * (416 + CAP)))
        out = os.path.join(out_dir, f"viz_{cat}.png")
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
