#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def summarize_dir(images_dir: Path) -> dict[str, object]:
    files = sorted(p for p in images_dir.iterdir() if p.is_file()) if images_dir.exists() else []
    if not files:
        return {
            "count": 0,
            "unique_sizes": [],
            "min_size": 0,
            "max_size": 0,
            "top_size": 0,
            "top_count": 0,
            "suspicious": True,
            "reason": "missing_or_empty",
        }

    sizes = [p.stat().st_size for p in files]
    freq: dict[int, int] = {}
    for size in sizes:
        freq[size] = freq.get(size, 0) + 1
    top_size, top_count = max(freq.items(), key=lambda kv: kv[1])
    unique_sizes = sorted(freq)

    suspicious = len(unique_sizes) == 1 or top_count / len(files) >= 0.95
    reason = "constant_or_near_constant_size" if suspicious else "ok"
    return {
        "count": len(files),
        "unique_sizes": unique_sizes,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "top_size": top_size,
        "top_count": top_count,
        "suspicious": suspicious,
        "reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify that round1 public_set images were restored cleanly after P0."
    )
    ap.add_argument(
        "--dataset_root",
        required=True,
        help="Path to Dataset/VAI_NVS_DATA/phase1/public_set",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional CSV output path.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    scene_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())

    headers = [
        "scene",
        "split",
        "count",
        "min_size",
        "max_size",
        "top_size",
        "top_count",
        "unique_size_count",
        "suspicious",
        "reason",
    ]

    rows: list[dict[str, object]] = []
    failing = []
    for scene_dir in scene_dirs:
        for split in ("train/images", "test/images"):
            result = summarize_dir(scene_dir / split)
            row = {
                "scene": scene_dir.name,
                "split": split,
                "count": result["count"],
                "min_size": result["min_size"],
                "max_size": result["max_size"],
                "top_size": result["top_size"],
                "top_count": result["top_count"],
                "unique_size_count": len(result["unique_sizes"]),
                "suspicious": "yes" if result["suspicious"] else "no",
                "reason": result["reason"],
            }
            rows.append(row)
            status = "FAIL" if result["suspicious"] else "OK"
            print(
                f"[{status}] {scene_dir.name:>8} {split:<12} "
                f"count={row['count']:<4} unique={row['unique_size_count']:<4} "
                f"top={row['top_size']} x {row['top_count']} "
                f"range=[{row['min_size']}, {row['max_size']}]"
            )
            if result["suspicious"]:
                failing.append(f"{scene_dir.name}:{split}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {out_csv}")

    if failing:
        print("\nP0 verify: FAIL")
        print("Still suspicious after restore:")
        for item in failing:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nP0 verify: PASS")
    print("No train/test image directory shows the previous constant-size truncation pattern.")


if __name__ == "__main__":
    main()
