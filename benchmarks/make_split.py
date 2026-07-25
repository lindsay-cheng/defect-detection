#!/usr/bin/env python3
"""stratified 80/20 train/val split for the YOLO defect dataset.

pair each image with its same-stem .txt label, group images by the FIRST class
index in their label (background/empty-label files group to class "good" by
convention since the dataset has none of those here), shuffle per group with
random.Random(seed), and *copy* (never move) into dataset/split/{train,val}.

usage:
    python make_split.py --src dataset/data --dst dataset/split --val-pct 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import random
import shutil
import sys
from pathlib import Path

NUM_CLASSES = 4
CLASS_NAMES = {0: "good", 1: "low_water", 2: "no_cap", 3: "no_label"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src", type=Path, required=True, help="source dir with images/ and labels/ subdirs"
    )
    p.add_argument("--dst", type=Path, required=True, help="destination split root")
    p.add_argument("--val-pct", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_first_class(label_path: Path) -> int:
    """return the FIRST class index in a YOLO label file.

    empty files (no annotations) are treated as background == class 0 (good).
    # ponytail: first-class grouping; upgrade = rarest-class or iterative
    # stratification once multi-class images exist (none in this dataset).
    """
    text = label_path.read_text().strip()
    if not text:
        return 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        return int(line.split()[0])
    return 0


def collect_classes(label_path: Path) -> set[int]:
    text = label_path.read_text().strip()
    if not text:
        return set()
    return {int(ln.split()[0]) for ln in text.splitlines() if ln.strip()}


def main() -> int:
    args = parse_args()
    src = args.src
    img_dir = src / "images"
    lbl_dir = src / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"ERROR: expected {img_dir} and {lbl_dir} to exist", file=sys.stderr)
        return 1

    images = sorted(img_dir.glob("*.jpg"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in lbl_dir.glob("*.txt")}

    unpaired_imgs = sorted(image_stems - label_stems)
    unpaired_lbls = sorted(label_stems - image_stems)

    groups: dict[int, list[Path]] = collections.defaultdict(list)
    multi_class_count = 0
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        first_cls = read_first_class(lbl)
        groups[first_cls].append(img)
        if len(collect_classes(lbl)) > 1:
            multi_class_count += 1
        if not (0 <= first_cls < NUM_CLASSES):
            print(
                f"ERROR: class index {first_cls} out of range 0..{NUM_CLASSES - 1} in {lbl}",
                file=sys.stderr,
            )
            return 1

    rng = random.Random(args.seed)
    for cls in groups:
        rng.shuffle(groups[cls])

    split_counts: dict[int, dict[str, int]] = {}
    move_log: list[tuple[str, str]] = []
    for cls, imgs in sorted(groups.items()):
        n = len(imgs)
        n_val = max(1, round(n * args.val_pct))
        n_train = n - n_val
        split_counts[cls] = {"train": n_train, "val": n_val}
        for img in imgs[:n_train]:
            move_log.append((img, "train"))
        for img in imgs[n_train:]:
            move_log.append((img, "val"))

    # clear + recreate dst
    dst = args.dst
    for sub in (
        dst / "train" / "images",
        dst / "train" / "labels",
        dst / "val" / "images",
        dst / "val" / "labels",
    ):
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)

    for img, split in move_log:
        lbl = lbl_dir / f"{img.stem}.txt"
        shutil.copy2(img, dst / split / "images" / img.name)
        shutil.copy2(lbl, dst / split / "labels" / lbl.name)

    # report
    print("=" * 56)
    print(f"make_split  src={src} dst={dst} val_pct={args.val_pct:.3f} seed={args.seed}")
    print("=" * 56)
    print(f"total images: {len(images)}   total labels: {len(label_stems)}")
    unpaired_img_tail = (
        ("  " + ", ".join(unpaired_imgs[:5]) + (" ..." if len(unpaired_imgs) > 5 else ""))
        if unpaired_imgs
        else ""
    )
    unpaired_lbl_tail = (
        ("  " + ", ".join(unpaired_lbls[:5]) + (" ..." if len(unpaired_lbls) > 5 else ""))
        if unpaired_lbls
        else ""
    )
    print(f"unpaired images (no label): {len(unpaired_imgs)}{unpaired_img_tail}")
    print(f"unpaired labels (no image): {len(unpaired_lbls)}{unpaired_lbl_tail}")
    print(f"multi-class images (contain >1 class): {multi_class_count}")
    print("-" * 56)
    print(f"{'class':<6} {'name':<14} {'train':>8} {'val':>8} {'total':>8}")
    print("-" * 56)
    tot_train = tot_val = 0
    for cls in sorted(split_counts):
        sc = split_counts[cls]
        tot = sc["train"] + sc["val"]
        tot_train += sc["train"]
        tot_val += sc["val"]
        name = CLASS_NAMES.get(cls, "?")
        print(f"{cls:<6} {name:<14} {sc['train']:>8} {sc['val']:>8} {tot:>8}")
    print("-" * 56)
    print(f"{'TOTAL':<21} {tot_train:>8} {tot_val:>8} {tot_train + tot_val:>8}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
