#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

from colmap_read_model import qvec2rotmat, read_cameras_binary, read_images_binary


def read_test_poses(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "image": Path(r["image_name"]).stem,
                    "qvec": np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                    "tvec": np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                }
            )
    return rows


def camera_center_and_forward(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_wc = qvec2rotmat(qvec)
    center = -rot_wc.T @ tvec
    forward = rot_wc.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    return center, forward


def load_flagged_proxy_rows(proxy_csv: Path) -> list[dict]:
    with open(proxy_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(float(r.get("refine_flag", "0") or 0)) == 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_csv", required=True)
    ap.add_argument("--test_poses_csv", required=True)
    ap.add_argument("--sparse_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--top_k", type=int, default=4)
    args = ap.parse_args()

    proxy_csv = Path(args.proxy_csv)
    test_poses_csv = Path(args.test_poses_csv)
    sparse_dir = Path(args.sparse_dir)
    out_csv = Path(args.out_csv)

    flagged_rows = load_flagged_proxy_rows(proxy_csv)
    if not flagged_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "refine_flag", "source_test_image", "vote_weight"])
        print("Flagged test images: 0")
        print("Mapped train images: 0")
        print("Saved:", out_csv)
        return

    test_pose_rows = read_test_poses(test_poses_csv)
    test_pose_map = {r["image"]: r for r in test_pose_rows}

    _ = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    train_images = read_images_binary(str(sparse_dir / "images.bin"))

    train_views = []
    for image in train_images.values():
        center, forward = camera_center_and_forward(image.qvec, image.tvec)
        train_views.append(
            {
                "image": Path(image.name).stem,
                "center": center,
                "forward": forward,
            }
        )

    vote_weights: dict[str, float] = {}
    for row in flagged_rows:
        test_image = row["image"]
        if test_image not in test_pose_map:
            continue
        test_pose = test_pose_map[test_image]
        test_center, test_forward = camera_center_and_forward(test_pose["qvec"], test_pose["tvec"])
        scored = []
        for train in train_views:
            dist = float(np.linalg.norm(train["center"] - test_center))
            align = float(np.clip(np.dot(train["forward"], test_forward), -1.0, 1.0))
            score = dist + 5.0 * (1.0 - align)
            scored.append((score, train["image"]))
        scored.sort(key=lambda item: (item[0], item[1]))
        top = scored[: max(1, args.top_k)]
        for rank, (_, train_image) in enumerate(top, start=1):
            vote_weights[train_image] = vote_weights.get(train_image, 0.0) + 1.0 / rank

    mapped = sorted(vote_weights.items(), key=lambda item: (-item[1], item[0]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "refine_flag", "source_test_image", "vote_weight"])
        for image, weight in mapped:
            writer.writerow([image, 1, "mapped_from_flagged_test_views", f"{weight:.6f}"])

    print("Flagged test images:", len(flagged_rows))
    print("Mapped train images:", len(mapped))
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
