"""LoRA SFT of Qwen3.5-2B to emit grasps as text (Unsloth Qwen3.5 vision recipe).

Example (our full method: endpoint text, medoid target, self-grounding-filtered labels):
  python train.py --data data/gapp5k --images data/gapp5k/images \
      --take ground --format endpoints --target medoid --epochs 2 --out runs/ep_ground
Smoke/overfit test:  --limit-objects 32 --epochs 30
Speed test:          --max-steps 60   (timing.json reports steady-state s/sample after warm-up)
"""
import argparse
import json
import os
import time

from unsloth import FastVisionModel  # must be imported before transformers/trl
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from grasp.data import SFTDataset, load_rows, train_items
from grasp.formats import FORMATS


class Timing(TrainerCallback):
    """Steady-state step time, skipping the first `warmup` steps (Triton kernel compilation)."""

    def __init__(self, samples_per_step, warmup, path):
        self.sps, self.warmup, self.path, self.times, self.t0 = samples_per_step, warmup, path, [], None

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kw):
        self.times.append(time.time() - self.t0)
        steady = self.times[self.warmup:]
        if steady and (state.global_step % 10 == 0 or state.global_step == state.max_steps):
            s = sum(steady) / len(steady)
            out = {"steps": len(self.times), "warmup_steps": self.warmup, "first_step_s": round(self.times[0], 2),
                   "steady_s_per_step": round(s, 3), "steady_s_per_sample": round(s / self.sps, 3)}
            json.dump(out, open(self.path, "w"), indent=2)
            print("timing", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="subset dir with raw.jsonl / clean.jsonl / ground.jsonl")
    ap.add_argument("--images", required=True, help="dir with extracted images")
    ap.add_argument("--take", choices=["raw", "clean", "ground"], default="ground")
    ap.add_argument("--format", choices=list(FORMATS), default="endpoints")
    ap.add_argument("--target", choices=["medoid", "top_score", "random"], default="medoid")
    ap.add_argument("--model", default="unsloth/Qwen3.5-2B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit-objects", type=int)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=416)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--print-modules", action="store_true", help="print module names/types of layer 0-3 and exit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)

    model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=False,
                                                       use_gradient_checkpointing="unsloth")
    if args.print_modules:
        for name, m in model.named_modules():
            if ".layers." not in name or any(f".layers.{i}." in name for i in range(4)):
                print(name, type(m).__name__)
        return

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # frozen vision encoder (small data, saves compute)
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none", random_state=args.seed,
    )
    model.print_trainable_parameters()

    fmt = FORMATS[args.format]
    items = train_items(load_rows(args.data, args.take, "train"), args.epochs, args.target, args.seed, args.limit_objects)
    train_ds = SFTDataset(items, args.images, fmt)
    print(f"train samples: {len(train_ds)}  example target: {train_ds[0]['messages'][1]['content'][0]['text']}")

    FastVisionModel.for_training(model)
    bf16 = is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor, resize=(args.image_size, args.image_size),
                                                chat_template_kwargs={"enable_thinking": False}),
        train_dataset=train_ds,
        callbacks=[Timing(args.bs * args.grad_accum, warmup=10, path=os.path.join(args.out, "timing.json"))],
        args=SFTConfig(
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=1,  # epochs are materialized in train_items (fresh prompt draw per epoch)
            max_steps=args.max_steps,
            warmup_ratio=0.03,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.001,
            fp16=not bf16, bf16=bf16,
            logging_steps=10,
            save_strategy="no",
            seed=args.seed,
            output_dir=args.out,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=512,
        ),
    )
    stats = trainer.train()
    model.save_pretrained(os.path.join(args.out, "adapter"))
    processor.save_pretrained(os.path.join(args.out, "adapter"))
    json.dump({"train_runtime_s": stats.metrics.get("train_runtime"), "train_loss": stats.metrics.get("train_loss"),
               "log_history": trainer.state.log_history}, open(os.path.join(args.out, "train_stats.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
