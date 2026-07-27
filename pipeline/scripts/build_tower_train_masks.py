#!/usr/bin/env python3
"""Sinh mask nhi phan vung anten/tru cho ANH TRAIN, dung de train.py cong them
1 loss phu trong so cao rieng vung nay (stage 2 cua antenna-focus 2-stage).

Chieu 8 dinh cua tower_bbox3d.json qua pose THAT cua tung anh trong
colmap/dense/sparse/0 (dung nguon reconstruction ma 03_train_3dgs.sh dung de
train khi SOURCE_MODE=prepared), lay convex hull 2D, ve mask + dilate bien
(cung phuong phap va cung tham so mac dinh voi trick/scripts/bootstrap_tower_masks.py
dang dung cho eval tower-crop cua M0, de nhat quan).

Neu 1 anh khong nhin thay du >=3 dinh bbox (bbox nam ngoai/sau camera), ghi
mask toan den (0) cho anh do -> stage 2 se khong cong them loss antenna cho
anh do (an toan, khong lam gi ca).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from colmap_read_model import qvec2rotmat, read_cameras_binary, read_images_binary


def project_points(corners: np.ndarray, qvec: np.ndarray, tvec: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    R = qvec2rotmat(qvec)
    cam = (R @ corners.T).T + tvec
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(corners), 2), dtype=np.float64)
    uv[valid, 0] = fx * (cam[valid, 0] / z[valid]) + cx
    uv[valid, 1] = fy * (cam[valid, 1] / z[valid]) + cy
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


def clamp_polygon(poly, width: int, height: int) -> list[tuple[int, int]]:
    out = []
    for x, y in poly:
        xi = min(max(int(round(x)), 0), width - 1)
        yi = min(max(int(round(y)), 0), height - 1)
        out.append((xi, yi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse_dir", required=True, help="vd: pipeline/work/<scene>/colmap/dense/sparse/0")
    ap.add_argument("--tower_bbox3d_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dilate_px", type=int, default=12)
    args = ap.parse_args()

    sparse_dir = Path(args.sparse_dir)
    cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    images = read_images_binary(str(sparse_dir / "images.bin"))

    bbox_path = Path(args.tower_bbox3d_json)
    if not bbox_path.exists():
        raise SystemExit(f"Không thấy {bbox_path}")
    bbox = json.loads(bbox_path.read_text(encoding="utf-8"))
    corners = np.asarray(bbox["corners"], dtype=np.float64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    n_empty = 0
    for image in images.values():
        cam = cameras[image.camera_id]
        if cam.model != "PINHOLE":
            raise SystemExit(
                f"Camera model={cam.model}, script này chỉ hỗ trợ PINHOLE (đúng output của "
                "pycolmap.undistort_images trong prepare_round1_scene.py)."
            )
        fx, fy, cx, cy = cam.params[:4]
        stem = Path(image.name).stem

        uv, valid = project_points(corners, image.qvec, image.tvec, fx, fy, cx, cy)
        valid_uv = [(float(x), float(y)) for (x, y), ok in zip(uv, valid) if ok]

        mask = Image.new("L", (cam.width, cam.height), 0)
        coverage = 0.0
        if len(valid_uv) >= 3:
            hull = convex_hull(valid_uv)
            if len(hull) >= 3:
                draw = ImageDraw.Draw(mask)
                draw.polygon(clamp_polygon(hull, cam.width, cam.height), fill=255)
                if args.dilate_px > 0:
                    x0 = max(min(p[0] for p in hull) - args.dilate_px, 0)
                    y0 = max(min(p[1] for p in hull) - args.dilate_px, 0)
                    x1 = min(max(p[0] for p in hull) + args.dilate_px, cam.width - 1)
                    y1 = min(max(p[1] for p in hull) + args.dilate_px, cam.height - 1)
                    draw.rectangle((x0, y0, x1, y1), outline=255, width=args.dilate_px)
                coverage = float((np.asarray(mask, dtype=np.uint8) > 0).mean())

        if coverage <= 0.0:
            n_empty += 1

        mask.save(out_dir / f"{stem}.png")
        summary_rows.append((stem, coverage))

    summary_csv = out_dir / "_train_mask_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask_coverage"])
        writer.writerows(summary_rows)

    print(f"Tổng ảnh: {len(summary_rows)} | mask rỗng (bbox không thấy): {n_empty}")
    print(f"Out dir: {out_dir}")
    print(f"Summary: {summary_csv}")
    if n_empty == len(summary_rows):
        raise SystemExit(
            "TOÀN BỘ mask đều rỗng -> tower_bbox3d.json có thể sai hệ toạ độ so với "
            f"{sparse_dir}. Dừng lại, đừng chạy stage 2 với mask rỗng."
        )


if __name__ == "__main__":
    main()
