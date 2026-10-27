# Grasping in the Language of Grounding

Language-driven grasp detection on [Grasp-Anything++](https://github.com/Fsoft-AIC/Grasp-Anything) with a small
general-purpose VLM (Qwen3.5-2B). Given an image and a prompt such as *"Lift highlighter by its marker."*, the model
predicts one rotated grasp rectangle `(x, y, w, h, θ)`.

The method reuses what the VLM already knows about pointing at things, twice:

1. **Grasp as native grounding text.** The grasp is written as its two jaw endpoints plus the jaw size, in Qwen's own
   0–1000 image frame, and learned with plain LoRA fine-tuning through the LM head (no new tokens, no regression head):

   ```
   p1, p2 = (x, y) ± (w/2)(cos θ, sin θ)        # canonical order: p1 has the smaller x
   target text: {"points_2d": [[x1, y1], [x2, y2]], "jaw": h}
   ```

   The conversion back to `(x, y, w, h, θ)` is exact, so evaluation uses the standard rectangle metric.
2. **Self-grounding label filter.** Before training, the same model (zero-shot) boxes every instance of the object
   named in the prompt. Objects whose ground-truth grasps fall outside those boxes are removed from the training set.
   About 22% of Grasp-Anything++ objects carry grasps on the wrong object; `data/check_alignment.py` traces these to the
   dataset's own segmentation masks.

## Results (5k-scene subset, 500 test scenes)

Success = IoU > 0.25 and angle error < 30° against any ground-truth grasp. *Verified* = the raw test split minus the
162 test objects that the filter flagged and that were confirmed mislabeled (`data/splits/verified_test.json`).
Two training seeds per LoRA row (mean).

| Method | Training labels | Raw test | Verified test |
|---|---|---|---|
| Constant grasp (center, median size, modal angle) | – | 13.3 | 13.8 |
| Qwen3.5-2B zero-shot (50 samples) | – | 10.0 | 11.1 |
| LoRA, `(x, y, w, h, θ)` as text | raw | 54.4 | 59.2 |
| LoRA, endpoint text | raw | 55.4 | 60.5 |
| **LoRA, endpoint text + self-grounding filter** | filtered | 55.4 | **61.9** |

These numbers come from a random 5k-scene subset with a scene-level split, so they are not comparable to results
on the official Grasp-Anything++ seen/unseen protocol.

## Repository layout

```
grasp/                 geometry (rect <-> endpoints, rotated IoU, metric), output formats, SFT dataset
data/build_subset.py   download a subset (HTTP range reads; no full 65 GB image download)
data/preprocess.py     raw.jsonl / clean.jsonl takes + training targets (medoid, top score)
ground.py              self-grounding filter, step 1 (GPU): zero-shot boxes for each object noun
data/ground_filter.py  self-grounding filter, step 2: keep/flag objects -> ground.jsonl take
train.py               LoRA SFT (Unsloth + TRL)
eval.py                generation + scoring on the test split
analysis/              baselines, verified-test scoring, failure analysis, dataset alignment check, paper figure
data/splits/           our exact scene split and the verified-test object list
tests/                 geometry / format unit tests (no GPU)
```

## Running it

Requirements: Python 3.10+, one CUDA GPU (we used a 16 GB T4; about 0.4 s per training sample, roughly 2 h per
run), and the packages in `requirements.txt` (then `unsloth install-kernels`). Data building and analysis run on CPU.

```bash
# 0. unit tests (CPU)
python tests/test_geometry.py

# 1. data: our exact 5k scenes (4000/500/500), then the raw and rule-based clean takes
python data/build_subset.py --out data/gapp5k --scenes data/splits/scenes.json
unzip -q data/gapp5k/images.zip -d data/gapp5k/images
python data/preprocess.py data/gapp5k

# 2. self-grounding label filter -> data/gapp5k/ground.jsonl
python ground.py --data data/gapp5k --images data/gapp5k/images --out runs/ground
python data/ground_filter.py data/gapp5k runs/ground --viz 12

# 3. train (full method) and evaluate
python train.py --data data/gapp5k --images data/gapp5k/images --take ground --format endpoints \
    --target medoid --epochs 2 --seed 3407 --out runs/ep_ground
python eval.py --data data/gapp5k --images data/gapp5k/images --adapter runs/ep_ground/adapter \
    --format endpoints --out runs/ep_ground/test

# 4. scores on raw / clean / verified test views (+ how often wrong labels are reproduced, degenerate outputs)
python analysis/score_subset.py data/gapp5k data/splits/verified_test.json ours=runs/ep_ground/test
```

Baselines and ablations use the same two scripts:

| Row | `train.py` flags | `eval.py` flags |
|---|---|---|
| zero-shot | – | no `--adapter`, `--format xywht --limit 50` |
| `(x, y, w, h, θ)` text | `--take raw --format xywht` | `--format xywht` |
| endpoint text, raw labels | `--take raw` | |
| rule-based clean labels | `--take clean` | |
| random / top-score target | `--target random` / `--target top_score` | |
| box-first | `--format endpoints_box` | `--format endpoints_box` |
| second seed | `--seed 1` | |

Analysis (CPU):

```bash
python analysis/constant_baseline.py data/gapp5k data/splits/verified_test.json
python analysis/analyze.py data/gapp5k a=runs/ep_ground/test b=runs/ep_raw/test     # failure modes, paired test
python analysis/viz_failures.py data/gapp5k runs/ep_raw/test --out-dir runs/viz      # position vs size of misses
python data/audit_labels.py data/gapp5k/annotations.jsonl                           # label-noise statistics
python data/check_alignment.py --data data/gapp5k --out runs/align                  # dataset masks vs labels
python analysis/alignment_summary.py data/gapp5k runs/align
python analysis/fig_label_noise.py data/gapp5k paper/figures/label_noise.png
```

## Notes

- Images come from `airvlab/Grasp-Anything` and annotations from `airvlab/Grasp-Anything-pp` on Hugging Face; files are
  joined by `<scene>_<obj>_<part>`, the same pairing as the official LGD loader.
- The vision encoder is frozen; LoRA (r = 16) covers all language-model projections, including Qwen3.5's Gated
  DeltaNet layers (16.8M of 2.23B parameters trainable). Decoding is greedy.
- Two runs with different seeds differ by up to ~1.6 points, so smaller differences should be read as ties.
