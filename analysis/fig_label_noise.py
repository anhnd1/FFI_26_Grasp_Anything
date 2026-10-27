"""Paper figure: Grasp-Anything++ annotation noise, with the zero-shot Qwen3.5 box for the named object.

Each panel shows the prompt (top), the dataset's grasps for the prompted object (green) and the box zero-shot
Qwen3.5-2B draws for the object noun (blue, from ground.py / data/ground_filter.py), with an in-image legend.

Usage: python analysis/fig_label_noise.py data/gapp5k paper/figures/label_noise.png [<scene>_<obj> ...]
"""
import collections
import io
import json
import os
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grasp.geometry import corners  # noqa: E402

# "Lift keychain by its ring." (grasps on a bottle opener), "Lift lime by its seeds." (grasps on an avocado)
PAPER_KEYS = ["33ceff42dff96d34307a4ca40c4e017ae0eb0058ffcb96ba7aa4d4179b23d60e_1",
              "1abcac212d58256c452d009802ee629e4a74542eeaeb0a2f149345e7e38c78a0_0"]
SIZE, GAP, BAND = 416, 8, 30
GREEN, BLUE = (0, 230, 70), (30, 130, 255)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def band(img, y, text_parts, f):
    """Dark translucent strip at height y with (text, color) parts written left to right."""
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).rectangle([0, y, SIZE, y + BAND], fill=(0, 0, 0, 170))
    img.alpha_composite(over)
    d, x = ImageDraw.Draw(img), 8
    for text, color in text_parts:
        if color is not None and text == "■":
            d.rectangle([x, y + 9, x + 16, y + BAND - 9], outline=color, width=3)
            x += 24
            continue
        d.text((x, y + 5), text, font=f, fill=color or "white")
        x += d.textlength(text, font=f) + 14


def main():
    subset, out = sys.argv[1:3]
    keys = sys.argv[3:] or PAPER_KEYS
    qwen = {f"{v['scene']}_{v['obj']}": v["boxes"]
            for v in map(json.loads, open(os.path.join(subset, "ground_verdicts.jsonl")))}
    rows = collections.defaultdict(list)
    for r in map(json.loads, open(os.path.join(subset, "raw.jsonl"), encoding="utf-8")):
        rows[f"{r['scene']}_{r['obj']}"].append(r)

    f = font(17)
    z = zipfile.ZipFile(os.path.join(subset, "images.zip"))
    sheet = Image.new("RGB", (len(keys) * SIZE + (len(keys) - 1) * GAP, SIZE), "white")
    for i, k in enumerate(keys):
        img = Image.open(io.BytesIO(z.read(rows[k][0]["image"]))).convert("RGBA")
        d = ImageDraw.Draw(img)
        for b in qwen.get(k, []):
            d.rectangle(b, outline=BLUE, width=4)
        for r in rows[k]:
            for g in r["grasps"]:
                d.polygon(corners(*g), outline=GREEN, width=2)
        band(img, 0, [(f'"{rows[k][0]["prompt"].rstrip(".")}"', None)], f)
        band(img, SIZE - BAND, [("■", GREEN), ("Dataset grasps", None), ("■", BLUE), ("Qwen3.5 (zero-shot)", None)], f)
        sheet.paste(img.convert("RGB"), (i * (SIZE + GAP), 0))
        print(k, "|", rows[k][0]["prompt"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
