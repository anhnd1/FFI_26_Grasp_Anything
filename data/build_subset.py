"""Build a small Grasp-Anything++ subset without downloading the full 65 GB image archive.

Images live in airvlab/Grasp-Anything as ONE zip64 split into two raw halves
(image_part_aa + image_part_ab). We treat the halves as a single virtual file served by
HTTP range requests, read only the central directory, then fetch just the sampled images.
Annotation zips (instructions .pkl, positive grasps .pt) are small enough to download whole.

Outputs (in --out):
    images.zip          sampled images, stored (no compression), named <scene>.<ext>
    annotations.jsonl   one line per (scene, object, part): id, scene, image, prompt, grasps, split
    stats.json          counts + raw-format inspection (grasp column ranges, prompt types)

Usage:
    python data/build_subset.py --out data/gapp5k                                   # random 5k scenes, seed 0
    python data/build_subset.py --out data/gapp5k --scenes data/splits/scenes.json  # exactly our subset/splits
    python data/build_subset.py --probe-images 5      # only test remote image fetching; no unpickling
"""
import argparse
import io
import json
import os
import pickle
import random
import struct
import threading
import time
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
IMAGE_PARTS = [HF.format(repo="airvlab/Grasp-Anything", file=f) for f in ("image_part_aa", "image_part_ab")]
ANNOT_REPO = "airvlab/Grasp-Anything-pp"
INSTR_ZIP, POS_ZIP = "grasp_instructions.zip", "grasp_label_positive.zip"


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view over several HTTP files concatenated end to end."""

    def __init__(self, urls):
        self._local = threading.local()
        self.parts, off = [], 0
        for u in urls:
            cdn, size = self._resolve(u)
            self.parts.append({"start": off, "size": size, "url": u, "cdn": cdn})
            off += size
        self.size, self.pos = off, 0

    def _session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def _resolve(self, url):
        # Follow HF's redirect once and cache the signed CDN url (CDN hits don't count toward hub rate limits).
        r = self._session().get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        r.raise_for_status()
        return r.url, int(r.headers["Content-Range"].split("/")[-1])

    def _get(self, part, a, b):
        for attempt in range(6):
            try:
                r = self._session().get(part["cdn"], headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code in (401, 403):  # signed url expired
                    part["cdn"], _ = self._resolve(part["url"])
                    continue
                r.raise_for_status()
                if len(r.content) != b - a + 1:
                    raise IOError(f"short read {len(r.content)} != {b - a + 1}")
                return r.content
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def pread(self, off, n):
        """Thread-safe read of n bytes at absolute offset off (may span parts)."""
        out, end = [], min(off + n, self.size)
        while off < end:
            p = next(p for p in self.parts if p["start"] <= off < p["start"] + p["size"])
            a = off - p["start"]
            b = min(p["size"], end - p["start"]) - 1
            out.append(self._get(p, a, b))
            off += b - a + 1
        return b"".join(out)

    # io interface used by zipfile (main thread only)
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else n
        data = self.pread(self.pos, n)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch_member(rf, zi):
    """Fetch one zip member with a single range request (local header + data), verify CRC."""
    want = 30 + len(zi.orig_filename.encode()) + 256 + zi.compress_size
    buf = rf.pread(zi.header_offset, want)
    if buf[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {zi.filename}")
    n, m = struct.unpack("<HH", buf[26:30])
    start = 30 + n + m
    if len(buf) < start + zi.compress_size:
        buf += rf.pread(zi.header_offset + len(buf), start + zi.compress_size - len(buf))
    data = buf[start:start + zi.compress_size]
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif zi.compress_type != zipfile.ZIP_STORED:
        raise IOError(f"unsupported compression {zi.compress_type} for {zi.filename}")
    if zlib.crc32(data) != zi.CRC:
        raise IOError(f"CRC mismatch for {zi.filename}")
    return data


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def open_image_index():
    t = time.time()
    rf = RemoteFile(IMAGE_PARTS)
    z = zipfile.ZipFile(rf)
    images = {stem(zi.filename): zi for zi in z.infolist() if not zi.is_dir()}
    print(f"image index: {len(images)} images, {rf.size / 1e9:.1f} GB virtual, {time.time() - t:.0f}s", flush=True)
    return rf, images


def fetch_images(rf, infos, dest_zip, workers):
    t, done = time.time(), 0
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_STORED) as out, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_member, rf, zi): zi for zi in infos}
        for f in as_completed(futs):
            zi = futs[f]
            out.writestr(os.path.basename(zi.filename), f.result())
            done += 1
            if done % 1000 == 0 or done == len(infos):
                print(f"  images {done}/{len(infos)} ({time.time() - t:.0f}s)", flush=True)


def jsonable(x):
    if hasattr(x, "tolist"):  # torch / numpy
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def describe(x):
    s = f"{type(x).__name__}"
    if hasattr(x, "shape"):
        s += f" shape={tuple(x.shape)} dtype={x.dtype}"
    return s + f" value={repr(x)[:300]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gapp5k")
    ap.add_argument("--tmp", default="data/.cache", help="where the two annotation zips are downloaded (~5.5 GB)")
    ap.add_argument("--scenes", help="JSON {train: [...], val: [...], test: [...]} of scene ids; overrides sampling")
    ap.add_argument("--n-scenes", type=int, default=5_000)
    ap.add_argument("--split", default="4000,500,500", help="train,val,test scene counts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--probe-images", type=int, default=0, help="only fetch N images and exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rf, images = open_image_index()
    if args.probe_images:
        infos = random.Random(args.seed).sample(list(images.values()), args.probe_images)
        fetch_images(rf, infos, os.path.join(args.out, "probe_images.zip"), args.workers)
        return

    import torch
    from huggingface_hub import hf_hub_download

    zips = {}
    for f in (INSTR_ZIP, POS_ZIP):
        t = time.time()
        path = hf_hub_download(ANNOT_REPO, f, repo_type="dataset", cache_dir=args.tmp)
        zips[f] = zipfile.ZipFile(path)
        print(f"{f}: downloaded+indexed in {time.time() - t:.0f}s", flush=True)

    instr = {stem(n): n for n in zips[INSTR_ZIP].namelist() if n.endswith(".pkl")}
    pos = {stem(n): n for n in zips[POS_ZIP].namelist() if n.endswith(".pt")}
    by_scene = defaultdict(list)
    for key in instr.keys() & pos.keys():
        scene = key.split("_")[0]
        if scene in images:
            by_scene[scene].append(key)
    print(f"instructions={len(instr)} positives={len(pos)} usable scenes={len(by_scene)}", flush=True)

    if args.scenes:
        fixed = json.load(open(args.scenes))
        scenes = [s for sp in ("train", "val", "test") for s in fixed[sp] if s in by_scene]
        split_of = {s: sp for sp in ("train", "val", "test") for s in fixed[sp]}
    else:
        n_tr, n_va, n_te = map(int, args.split.split(","))
        assert n_tr + n_va + n_te == args.n_scenes
        scenes = sorted(by_scene)
        random.Random(args.seed).shuffle(scenes)
        scenes = scenes[:args.n_scenes]
        split_of = {s: "train" if i < n_tr else "val" if i < n_tr + n_va else "test" for i, s in enumerate(scenes)}

    print("fetching images...", flush=True)
    fetch_images(rf, [images[s] for s in scenes], os.path.join(args.out, "images.zip"), args.workers)

    stats = {"usable_scenes": len(by_scene), "splits": defaultdict(lambda: {"scenes": 0, "pairs": 0}),
             "samples": [], "grasp_rows": 0, "grasp_cols_min": None, "grasp_cols_max": None, "prompt_types": defaultdict(int)}
    with open(os.path.join(args.out, "annotations.jsonl"), "w", encoding="utf-8") as fo:
        for s in scenes:
            sp = split_of[s]
            stats["splits"][sp]["scenes"] += 1
            for key in sorted(by_scene[s]):
                prompt = pickle.loads(zips[INSTR_ZIP].read(instr[key]))
                grasps = torch.load(io.BytesIO(zips[POS_ZIP].read(pos[key])), weights_only=True)
                if len(stats["samples"]) < 5:
                    stats["samples"].append({"key": key, "prompt": describe(prompt), "grasps": describe(grasps)})
                stats["prompt_types"][type(prompt).__name__] += 1
                g = jsonable(grasps)
                rows = [r for r in g if isinstance(r, list)] if isinstance(g, list) else []
                if rows and all(isinstance(v, (int, float)) for v in rows[0]):
                    stats["grasp_rows"] += len(rows)
                    cols = list(zip(*rows))
                    lo, hi = [min(c) for c in cols], [max(c) for c in cols]
                    if stats["grasp_cols_min"] is None or len(stats["grasp_cols_min"]) != len(lo):
                        stats["grasp_cols_min"], stats["grasp_cols_max"] = lo, hi
                    else:
                        stats["grasp_cols_min"] = [min(a, b) for a, b in zip(stats["grasp_cols_min"], lo)]
                        stats["grasp_cols_max"] = [max(a, b) for a, b in zip(stats["grasp_cols_max"], hi)]
                ext = os.path.splitext(images[s].filename)[1]
                fo.write(json.dumps({"id": key, "scene": s, "image": s + ext, "prompt": jsonable(prompt),
                                     "grasps": g, "split": sp}) + "\n")
                stats["splits"][sp]["pairs"] += 1

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
