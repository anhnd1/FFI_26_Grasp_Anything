"""Self-grounding label filter, step 1 (GPU): zero-shot Qwen3.5 boxes every instance of each object's name.

One query per object (all its part prompts name the same object): "Locate every <noun> ...". The raw answer and every
parsed bbox_2d are saved; the keep/flag decision is made locally by data/ground_filter.py, so thresholds and the
coordinate convention can be checked without the GPU.

Usage: python ground.py --data D --images I --splits train,val,test [--shard 0/2] --out O
"""
import argparse
import json
import os
import re
import time

from unsloth import FastVisionModel
import torch
from PIL import Image

from grasp.data import load_rows, object_name

_THINK = re.compile(r"<think>.*?</think>", re.S)
_BOX = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]*)\]')


def instruction(noun):
    return (f"Locate every {noun} in the image. Output a JSON list with one entry per {noun}: "
            f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{noun}"}}].')


def parse_boxes(text):
    boxes = []
    for m in _BOX.finditer(text):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if len(nums) == 4:
            boxes.append(nums)
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles objects i, i+n, ...")
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    objs = {}
    for split in args.splits.split(","):
        for r in load_rows(args.data, "raw", split):
            objs.setdefault((r["scene"], r["obj"]), r)
    i, n = map(int, args.shard.split("/"))
    rows = [objs[k] for k in sorted(objs)][i::n]

    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=False)
    FastVisionModel.for_inference(model)
    processor.tokenizer.padding_side = "left"

    out_path = os.path.join(args.out, f"ground_{i}.jsonl")
    t0 = time.time()
    with open(out_path, "w") as f:
        for b in range(0, len(rows), args.bs):
            batch = rows[b:b + args.bs]
            images = [Image.open(os.path.join(args.images, r["image"])).convert("RGB") for r in batch]
            nouns = [object_name(r["prompt"]) for r in batch]
            texts = [processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction(nn)}]}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False) for img, nn in zip(images, nouns)]
            inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=True, do_sample=False)
            answers = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, img, nn, a in zip(batch, images, nouns, answers):
                a = _THINK.sub("", a).strip()
                f.write(json.dumps({"scene": r["scene"], "obj": r["obj"], "split": r["split"], "noun": nn,
                                    "image_size": img.size, "text": a, "boxes": parse_boxes(a)}) + "\n")
            if b == 0:
                print("first answers:", answers[:3], flush=True)
            print(f"{b + len(batch)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    print("done", out_path)


if __name__ == "__main__":
    main()
