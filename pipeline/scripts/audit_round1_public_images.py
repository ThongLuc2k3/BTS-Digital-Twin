#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def audit_scene(scene_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train/images", "test/images"):
        images_dir = scene_dir / split
        files = sorted(p for p in images_dir.iterdir() if p.is_file()) if images_dir.exists() else []
        if not files:
            rows.append({
                "scene": scene_dir.name,
                "split": split,
                "count": 0,
                "unique_sizes": "",
                "min_size": "",
                "max_size": "",
                "constant_size_count": 0,
                "constant_size_value": "",
                "suspicious": "missing",
            })
            continue

        sizes = [p.stat().st_size for p in files]
        counts: dict[int, int] = {}
        for size in sizes:
            counts[size] = counts.get(size, 0) + 1
        top_size, top_count = max(counts.items(), key=lambda kv: kv[1])
        unique_sizes = sorted(counts)
        suspicious = len(unique_sizes) == 1 or top_count / len(files) >= 0.95

        rows.append({
            "scene": scene_dir.name,
            "split": split,
            "count": len(files),
            "unique_sizes": ",".join(str(x) for x in unique_sizes[:10]),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "constant_size_count": top_count,
            "constant_size_value": top_size,
            "suspicious": "yes" if suspicious else "no",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit image file sizes in round1 public_set and flag suspicious constant-size truncation."
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
    rows: list[dict[str, object]] = []
    for scene_dir in scene_dirs:
        rows.extend(audit_scene(scene_dir))

    headers = [
        "scene",
        "split",
        "count",
        "unique_sizes",
        "min_size",
        "max_size",
        "constant_size_count",
        "constant_size_value",
        "suspicious",
    ]

    for row in rows:
        print(
            f"{row['scene']:>8}  {row['split']:<12}  count={row['count']:<4}  "
            f"unique=[{row['unique_sizes']}]  top={row['constant_size_value']} x {row['constant_size_count']}  "
            f"suspicious={row['suspicious']}"
        )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
