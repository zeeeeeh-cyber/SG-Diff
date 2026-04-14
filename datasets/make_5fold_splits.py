#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from pathlib import Path
import random
from typing import Set, List

ID_RE = re.compile(r"^(\d{3})")  

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def collect_ids(folder: Path) -> Set[str]:
    ids = set()
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = ID_RE.match(p.name)
        if m:
            ids.add(m.group(1))
    return ids

def write_list(path: Path, ids: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in ids:
            f.write(x + "\n")

def main():
    ap = argparse.ArgumentParser("Make 5-fold splits for paired dataset.")
    ap.add_argument("--root", required=True, type=str, help="dataset root containing artifactIma/ and cleanIma/")
    ap.add_argument("--artifactIma", default="artifactIma", type=str, help="subfolder name for artifactIma (default: artifactIma)")
    ap.add_argument("--cleanIma", default="cleanIma", type=str, help="subfolder name for cleanIma (default: cleanIma)")
    ap.add_argument("--out", default="splits", type=str, help="output splits folder under root (default: splits)")
    ap.add_argument("--k", default=5, type=int, help="number of folds (default 5)")
    ap.add_argument("--seed", default=42, type=int, help="random seed (default 42)")
    ap.add_argument("--strict_id", action="store_true",
                    help="require ids. If not set, use intersection of two dirs.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    artifactIma_dir = root / args.artifactIma
    cleanIma_dir = root / args.cleanIma
    out_dir = root / args.out

    if not artifactIma_dir.is_dir():
        raise FileNotFoundError(f"artifactIma dir not found: {artifactIma_dir}")
    if not cleanIma_dir.is_dir():
        raise FileNotFoundError(f"cleanIma dir not found: {cleanIma_dir}")

    ids_a = collect_ids(artifactIma_dir)
    ids_b = collect_ids(cleanIma_dir)
    inter = sorted(ids_a & ids_b)

    if args.strict_id:
        expected = [f"{i:03d}" for i in range(1, 366)]
        missing = sorted(set(expected) - set(inter))
        extra = sorted(set(inter) - set(expected))
        if missing:
            raise RuntimeError(f"Missing paired ids in intersection. Missing count={len(missing)}. "
                               f"First 20: {missing[:20]}")
        if extra:
            raise RuntimeError(f"Found extra ids in intersection. Count={len(extra)}. First 20: {extra[:20]}")
        ids = expected
    else:
        ids = inter

    if len(ids) == 0:
        raise RuntimeError("No paired ids found (intersection is empty). Check filename prefixes like 001.png")

    rng = random.Random(args.seed)
    ids_shuf = ids[:]
    rng.shuffle(ids_shuf)

    fold_ids = [[] for _ in range(args.k)]
    for i, id3 in enumerate(ids_shuf):
        fold_ids[i % args.k].append(id3)

    out_dir.mkdir(parents=True, exist_ok=True)
    assign_csv = out_dir / "fold_assign.csv"
    with assign_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "fold"])
        for fold, lst in enumerate(fold_ids):
            for id3 in sorted(lst):
                w.writerow([id3, fold])

    for k in range(args.k):
        test_ids = sorted(fold_ids[k])
        train_ids = sorted([x for j in range(args.k) if j != k for x in fold_ids[j]])
        write_list(out_dir / f"fold{k}_test.txt", test_ids)
        write_list(out_dir / f"fold{k}_train.txt", train_ids)

    # summary
    print("ROOT:", root)
    print("artifactIma ids:", len(ids_a), "cleanIma ids:", len(ids_b), "intersection:", len(inter))
    print("Using ids:", len(ids), "k:", args.k, "seed:", args.seed)
    for k in range(args.k):
        print(f"fold{k}: test={len(fold_ids[k])} (ids {min(fold_ids[k])}..{max(fold_ids[k])} after shuffle/assign not meaningful)")
    print("Wrote:", assign_csv)
    print("Wrote fold{k}_train.txt / fold{k}_test.txt under:", out_dir)

if __name__ == "__main__":
    main()
