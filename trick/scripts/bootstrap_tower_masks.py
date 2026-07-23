#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
    ], dtype=np.float64)


def read_test_poses(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_name": r["image_name"],
                "qvec": np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                "tvec": np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                "fx": float(r["fx"]),
                "fy": float(r["fy"]),
                "cx": float(r["cx"]),
                "cy": float(r["cy"]),
                "width": int(float(r["width"])),
                "height": int(float(r["height"])),
            })
    return rows


def project_points(points_world: np.ndarray, pose: dict) -> tuple[np.ndarray, np.ndarray]:
    rot = qvec2rotmat(pose["qvec"])
    cam = (rot @ points_world.T).T + pose["tvec"]
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(points_world), 2), dtype=np.float64)
    uv[valid, 0] = pose["fx"] * (cam[valid, 0] / z[valid]) + pose["cx"]
    uv[valid, 1] = pose["fy"] * (cam[valid, 1] / z[valid]) + pose["cy"]
    return uv, valid


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def clamp_polygon(poly: list[tuple[float, float]], width: int, height: int) -> list[tuple[int, int]]:
    out = []
    for x, y in poly:
        xi = min(max(int(round(x)), 0), width - 1)
        yi = min(max(int(round(y)), 0), height - 1)
        out.append((xi, yi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--tower_bbox3d_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dilate_px", type=int, default=12)
    args = ap.parse_args()

    poses_csv = Path(args.dataset_root) / args.scene / "test" / "test_poses.csv"
    if not poses_csv.exists():
        raise SystemExit(f"Không thấy {poses_csv}")

    bbox_path = Path(args.tower_bbox3d_json)
    if not bbox_path.exists():
        raise SystemExit(f"Không thấy {bbox_path}")

    bbox = json.loads(bbox_path.read_text(encoding="utf-8"))
    corners = np.asarray(bbox["corners"], dtype=np.float64)
    poses = read_test_poses(poses_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for pose in poses:
        stem = Path(pose["image_name"]).stem
        uv, valid = project_points(corners, pose)
        valid_uv = [(float(x), float(y)) for (x, y), ok in zip(uv, valid) if ok]
        if len(valid_uv) < 3:
            continue

        hull = convex_hull(valid_uv)
        if len(hull) < 3:
            continue

        mask = Image.new("L", (pose["width"], pose["height"]), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(clamp_polygon(hull, pose["width"], pose["height"]), fill=255)
        if args.dilate_px > 0:
            x0 = max(min(p[0] for p in hull) - args.dilate_px, 0)
            y0 = max(min(p[1] for p in hull) - args.dilate_px, 0)
            x1 = min(max(p[0] for p in hull) + args.dilate_px, pose["width"] - 1)
            y1 = min(max(p[1] for p in hull) + args.dilate_px, pose["height"] - 1)
            draw.rectangle((x0, y0, x1, y1), outline=255, width=args.dilate_px)

        out_path = out_dir / f"{stem}.png"
        mask.save(out_path)

        arr = np.asarray(mask, dtype=np.uint8) > 0
        coverage = float(arr.mean())
        summary_rows.append((stem, coverage))

    summary_csv = out_dir / "_bootstrap_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask_coverage"])
        writer.writerows(summary_rows)

    if not summary_rows:
        raise SystemExit("Không tạo được bootstrap mask nào.")

    coverages = np.array([r[1] for r in summary_rows], dtype=np.float64)
    print(f"Bootstrap masks: {len(summary_rows)}")
    print(f"Coverage mean : {coverages.mean():.4f}")
    print(f"Coverage min  : {coverages.min():.4f}")
    print(f"Coverage max  : {coverages.max():.4f}")
    print(f"Saved dir     : {out_dir}")
    print(f"Saved summary : {summary_csv}")


if __name__ == "__main__":
    main()
