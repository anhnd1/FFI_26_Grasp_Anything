"""Samples for SFT and evaluation, built from raw.jsonl / clean.jsonl (data/preprocess.py).

Training epoch = every object once, with one of its part prompts drawn at random (part prompts mostly share
labels, so iterating over all ~2.8 prompts per object would mostly repeat the same target).
"""
import json
import os
import random
import re
from collections import defaultdict

from PIL import Image

# GA++ prompts are "<verb> <object> <at|on|by|...> its <part>." (checked: all 27,098 rows of the 5k subset match)
_PROMPT = re.compile(r"^(?:take hold of|pick up|grasp|grip|grab|lift|fetch|take|hold)\s+(.*?)\s+"
                     r"(?:at|on|by|from|using|with|via|through|near|around)\s+(?:its|the)\s+.+$", re.I)


def object_name(prompt):
    """Object noun of a GA++ prompt ("Grip pen on its cap." -> "pen"); None if the prompt doesn't parse."""
    m = _PROMPT.match(prompt.strip().rstrip("."))
    return m.group(1).lower() if m else None


def load_rows(subset_dir, take, split):
    path = os.path.join(subset_dir, f"{take}.jsonl")
    return [r for r in map(json.loads, open(path, encoding="utf-8")) if r["split"] == split]


def target_grasp(row, target, rng):
    if target == "random":
        return row["grasps"][rng.randrange(len(row["grasps"]))]
    return row["grasps"][row["target"][target]]


def user_content(image, prompt, fmt):
    return [{"type": "image", "image": image}, {"type": "text", "text": f"{prompt} {fmt.instruction}"}]


def train_items(rows, epochs, target, seed, limit_objects=None):
    """List of (row, target_grasp), epochs x objects, shuffled per epoch. Deterministic given seed."""
    rng = random.Random(seed)
    by_obj = defaultdict(list)
    for r in rows:
        by_obj[(r["scene"], r["obj"])].append(r)
    objs = sorted(by_obj)
    if limit_objects:
        objs = objs[:limit_objects]
    items = []
    for _ in range(epochs):
        order = objs[:]
        rng.shuffle(order)
        for o in order:
            row = rng.choice(by_obj[o])
            items.append((row, target_grasp(row, target, rng)))
    return items


class SFTDataset:
    """List-like dataset of chat samples for SFTTrainer + UnslothVisionDataCollator; images load lazily."""

    def __init__(self, items, image_dir, fmt):
        self.items, self.image_dir, self.fmt = items, image_dir, fmt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, g = self.items[i]
        image = Image.open(os.path.join(self.image_dir, row["image"])).convert("RGB")
        return {"messages": [
            {"role": "user", "content": user_content(image, row["prompt"], self.fmt)},
            {"role": "assistant", "content": [{"type": "text", "text": self.fmt.encode(g, row["grasps"])}]},
        ]}
