"""Generate grasps for a split and score them with the rectangle metric (IoU > 0.25 and angle error < 30 deg
against ANY ground-truth grasp of the sample).

Predicts once on the raw take and reports both views:
  raw    all samples of the split (labels as shipped by GA++)
  clean  the subset that survives data/preprocess.py (no degenerate / copied labels)

Zero-shot:   python eval.py --data D --images I --format endpoints --out O
Fine-tuned:  python eval.py --data D --images I --adapter RUN/adapter --format endpoints --out O
Consensus:   add --samples K [--temperature T]: sample K grasps per input and output their medoid (the same
             consensus rule used to pick training targets); default K=1 is greedy decoding.
"""
import argparse
import json
import os
import random
import re
import time

from unsloth import FastVisionModel
import torch
from PIL import Image

from grasp.data import load_rows, user_content
from grasp.formats import FORMATS
from grasp.geometry import is_success, medoid_index

_THINK = re.compile(r"<think>.*?</think>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--format", choices=list(FORMATS), default="endpoints")
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--adapter", help="LoRA adapter dir from train.py (omit for zero-shot)")
    ap.add_argument("--limit", type=int, help="random subset of N samples (seeded)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--samples", type=int, default=1, help="K > 1: sample K grasps and output their medoid")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_rows(args.data, "raw", args.split)
    clean_ids = {r["id"] for r in load_rows(args.data, "clean", args.split)}
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    fmt = FORMATS[args.format]

    model, processor = FastVisionModel.from_pretrained(args.adapter or args.model, load_in_4bit=False)
    FastVisionModel.for_inference(model)
    processor.tokenizer.padding_side = "left"

    preds, t0 = [], time.time()
    for i in range(0, len(rows), args.bs):
        batch = rows[i:i + args.bs]
        images = [Image.open(os.path.join(args.images, r["image"])).convert("RGB") for r in batch]
        texts = [processor.apply_chat_template([{"role": "user", "content": user_content(img, r["prompt"], fmt)}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                 for img, r in zip(images, batch)]
        inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(model.device)
        k = args.samples
        sampling = {"do_sample": True, "temperature": args.temperature, "num_return_sequences": k} if k > 1 else {"do_sample": False}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, use_cache=True, **sampling)
        texts_out = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, r in enumerate(batch):
            group = texts_out[j * k:(j + 1) * k]  # num_return_sequences keeps each input's samples contiguous
            cands = [c for c in (fmt.decode(_THINK.sub("", t).strip()) for t in group) if c is not None]
            pred = cands[medoid_index(cands)] if cands else None
            preds.append({"id": r["id"], "text": group[0] if k == 1 else group, "pred": pred,
                          "success": bool(pred) and is_success(pred, r["grasps"]), "clean": r["id"] in clean_ids})
        print(f"{len(preds)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    def rate(ps):
        return {"n": len(ps), "success": round(sum(p["success"] for p in ps) / max(1, len(ps)), 4),
                "parse_fail": round(sum(p["pred"] is None for p in ps) / max(1, len(ps)), 4)}

    metrics = {"split": args.split, "format": args.format, "adapter": args.adapter, "limit": args.limit,
               "samples": args.samples, "temperature": args.temperature if args.samples > 1 else None,
               "raw": rate(preds), "clean": rate([p for p in preds if p["clean"]]),
               "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
