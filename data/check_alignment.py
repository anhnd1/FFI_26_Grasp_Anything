"""Check that GA++ labels are attached to the right object (CPU only; downloads only the needed zip members).

For every object in our subset, compares three independent sources keyed by <scene>_<obj>[_<part>]:
  - Grasp-Anything v1 scene_description <scene>.pkl -> (scene prompt, object list); LGD's official GA++ loader
    takes the object name as queries[obj] from this file (utils/data/grasp_anywhere_data.py).
  - Grasp-Anything v1 mask.zip     per-object segmentation masks
  - GA++ part_mask.zip             per-(object, part) masks the GA++ grasps were generated from
and asks where the GA++ grasp centers (raw.jsonl) fall: in the object's own v1 mask, in another object's mask,
or nowhere. Mask orientation is not assumed: fractions are computed for mask[y, x] and mask[x, y].

Outputs (in --out): alignment.jsonl (one line per object), masks_meta.json (zip listings, array shapes), and
mask_bboxes.jsonl (bbox/area per mask; part masks also carry the fraction of their own grasps inside).

Usage: python data/check_alignment.py --data data/gapp5k --out runs/align
"""
import argparse
import io
import json
import os
import pickle
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_subset import HF, RemoteFile, fetch_member, stem  # noqa: E402

V1, PP = "airvlab/Grasp-Anything", "airvlab/Grasp-Anything-pp"
ZIPS = {"desc": (V1, "scene_description.zip"), "mask": (V1, "mask.zip"), "part": (PP, "part_mask.zip")}


def open_zip(repo, f):
    rf = RemoteFile([HF.format(repo=repo, file=f)])
    z = zipfile.ZipFile(rf)
    return rf, {stem(zi.filename): zi for zi in z.infolist() if not zi.is_dir()}


def load_any(name, data):
    if name.endswith(".npy"):
        try:
            return np.load(io.BytesIO(data), allow_pickle=False)
        except ValueError:
            return np.load(io.BytesIO(data), allow_pickle=True)
    if name.endswith(".pt"):
        import torch
        return torch.load(io.BytesIO(data), weights_only=False)
    return pickle.loads(data)


def as_mask(x):
    m = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
    m = np.squeeze(m)
    return m > 0 if m.ndim == 2 else None


def frac_inside(mask, centers, transpose):
    if mask is None or not centers:
        return None
    h, w = mask.shape
    hit = 0
    for x, y in centers:
        r, c = (int(round(x)), int(round(y))) if transpose else (int(round(y)), int(round(x)))
        hit += 0 <= r < h and 0 <= c < w and bool(mask[r, c])
    return round(hit / len(centers), 3)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(line) for line in open(os.path.join(args.data, "raw.jsonl"), encoding="utf-8")]
    objs = defaultdict(list)
    for r in rows:
        objs[(r["scene"], r["obj"])].append(r)
    scenes = sorted({s for s, _ in objs})

    t = time.time()
    remote = {k: open_zip(*v) for k, v in ZIPS.items()}
    meta = {k: {"members": len(idx), "first": sorted(idx)[:5]} for k, (_, idx) in remote.items()}
    print("indexed zips", json.dumps(meta), f"{time.time() - t:.0f}s", flush=True)

    scene_set = set(scenes)
    by_id = {r["id"]: r for r in rows}
    mask_idx, part_idx = remote["mask"][1], remote["part"][1]
    examples = {}

    def work(kind, key):
        """Fetch one member and reduce it at once (masks -> bool, part masks -> fractions), keeping memory small."""
        rf, idx = remote[kind]
        v = load_any(idx[key].filename, fetch_member(rf, idx[key]))
        if kind == "desc":
            return v
        if kind not in examples:
            arr = np.asarray(v.numpy() if hasattr(v, "numpy") else v)
            examples[kind] = {"key": key, "type": type(v).__name__, "shape": list(arr.shape), "dtype": str(arr.dtype),
                              "unique": [float(u) for u in np.unique(arr)[:5]]}
        m = as_mask(v)
        info = {"bbox": None if m is None else bbox(m), "area": None if m is None else int(m.sum())}
        if kind == "part":
            cs = [g[:2] for g in by_id[key]["grasps"]]
            info["frac_yx"], info["frac_T"] = frac_inside(m, cs, False), frac_inside(m, cs, True)
            return info
        return m, info

    want = [("desc", s) for s in scenes if s in remote["desc"][1]]
    want += [("mask", k) for k in mask_idx if k.split("_")[0] in scene_set]
    want += [("part", i) for i in by_id if i in part_idx]
    print(f"fetching {len(want)} members", flush=True)
    got = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (w, val) in enumerate(zip(want, ex.map(lambda w: work(*w), want))):
            got[w] = val
            if i % 2000 == 0:
                print(f"  {i}/{len(want)} {time.time() - t:.0f}s", flush=True)

    first = next((v for (k, _), v in got.items() if k == "desc"), None)
    meta["desc_example"] = repr(first)[:600]
    meta["examples"] = examples
    json.dump(meta, open(os.path.join(args.out, "masks_meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2), flush=True)

    masks_by_scene = defaultdict(dict)
    part_info = {}
    with open(os.path.join(args.out, "mask_bboxes.jsonl"), "w") as fb:
        for (kind, key), v in got.items():
            if kind == "desc":
                continue
            info = v[1] if kind == "mask" else v
            if kind == "mask":
                masks_by_scene[key.split("_")[0]][key.split("_", 1)[1]] = v[0]
            else:
                part_info[key] = info
            fb.write(json.dumps({"kind": kind, "key": key, **info}) + "\n")

    with open(os.path.join(args.out, "alignment.jsonl"), "w") as fo:
        for (s, o), rs in sorted(objs.items()):
            desc = got.get(("desc", s))
            queries = desc[1] if isinstance(desc, (list, tuple)) and len(desc) > 1 else None
            centers = [g[:2] for r in rs for g in r["grasps"]]
            rec = {"scene": s, "obj": o, "split": rs[0]["split"], "prompts": sorted({r["prompt"] for r in rs}),
                   "v1_queries": [str(q) for q in queries] if queries is not None else None,
                   "v1_name": str(queries[int(o)]) if queries is not None and int(o) < len(queries) else None,
                   "v1_masks": sorted(masks_by_scene[s])}
            for tr in (False, True):
                tag = "T" if tr else "yx"
                fr = {mo: frac_inside(m, centers, tr) for mo, m in masks_by_scene[s].items()}
                rec[f"own_mask_frac_{tag}"] = fr.get(o)
                rec[f"mask_fracs_{tag}"] = fr
                rec[f"part_mask_frac_{tag}"] = {r["part"]: part_info.get(r["id"], {}).get(f"frac_{tag}") for r in rs}
            fo.write(json.dumps(rec) + "\n")
    print("done", time.time() - t, flush=True)


if __name__ == "__main__":
    main()
