#!/usr/bin/env python3
"""Tự chấm PSNR/SSIM/LPIPS trên public_set (nơi DUY NHẤT có ảnh GT thật để so sánh).

Không dùng được cho private_set1 (không có ảnh GT) — mục đích của script này là
đánh giá pipeline TRƯỚC khi áp dụng cho private set, đúng lưu ý của BTC ở mục 4.7
đề bài ("kiểm tra kỹ pipeline trên public set trước khi chạy chính thức").

Cách dùng:
    python 05_eval_metrics.py --scene HCM0181
    python 05_eval_metrics.py --all_public

Yêu cầu đã chạy 04_render_test_poses.py cho scene đó trước (renders nằm ở
pipeline/work/<scene>/renders/<stem>.png).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene, all_scenes, Scene

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


def load_img01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def eval_scene(scene: Scene, renders_dir: Path, lpips_fn) -> list[tuple]:
    gt_dir = scene.gt_test_images_dir
    rows = []
    gt_paths = sorted(gt_dir.glob("*"))
    for gt_path in gt_paths:
        stem = gt_path.stem
        render_path = renders_dir / f"{stem}.png"
        if not render_path.exists():
            print(f"  [THIẾU] {stem}: không có render tại {render_path}")
            continue
        gt = load_img01(gt_path)
        pred = load_img01(render_path)
        if gt.shape != pred.shape:
            raise ValueError(f"{stem}: kích thước khác nhau GT={gt.shape[:2]} pred={pred.shape[:2]}")

        psnr_v = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_v = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)

        lpips_v = float("nan")
        if lpips_fn is not None:
            t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            if torch.cuda.is_available():
                t_gt, t_pred = t_gt.cuda(), t_pred.cuda()
            with torch.no_grad():
                lpips_v = float(lpips_fn(t_gt, t_pred).item())

        rows.append((stem, psnr_v, ssim_v, lpips_v))

    n_missing = len(gt_paths) - len(rows)
    if n_missing:
        print(f"  [CẢNH BÁO] {scene.name}: thiếu {n_missing}/{len(gt_paths)} render — "
              f"điểm trung bình dưới đây KHÔNG đại diện cho toàn bộ scene.")
    return rows


def write_csv(csv_path: Path, rows: list[tuple]) -> None:
    """Ghi điểm từng ảnh test ra CSV — dùng để vẽ biểu đồ/so sánh ảnh trong notebook
    (xem cell sau Bước 6 của kaggle_public.ipynb), vì console chỉ in mean/min/max."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "lpips"])
        writer.writerows(rows)


def print_stats(name: str, rows: list[tuple]):
    arr = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f"\n=== {name}: {len(rows)} ảnh ===")
    print(f"  PSNR  mean={arr[:, 0].mean():.3f}  min={arr[:, 0].min():.3f}")
    print(f"  SSIM  mean={arr[:, 1].mean():.4f}  min={arr[:, 1].min():.4f}")
    if not np.isnan(arr[:, 2]).all():
        print(f"  LPIPS mean={arr[:, 2].mean():.4f}  max={arr[:, 2].max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="1 scene public cụ thể, vd HCM0181")
    g.add_argument("--all_public", action="store_true")
    ap.add_argument("--renders_root", default=None, help="Mặc định pipeline/work")
    ap.add_argument("--no_lpips", action="store_true", help="Bỏ qua LPIPS (nếu chưa cài package lpips)")
    args = ap.parse_args()

    scenes = [get_scene(args.scene)] if args.scene else all_scenes("public")
    pipeline_root = Path(__file__).resolve().parents[1]
    renders_root = Path(args.renders_root) if args.renders_root else pipeline_root / "work"

    lpips_fn = None
    if not args.no_lpips:
        if not _HAS_LPIPS:
            print("[CẢNH BÁO] Chưa cài package `lpips` (pip install lpips) — bỏ qua LPIPS, chỉ tính PSNR/SSIM.")
        else:
            lpips_fn = lpips.LPIPS(net="alex")
            if torch.cuda.is_available():
                lpips_fn = lpips_fn.cuda()

    all_rows = []
    for scene in scenes:
        if scene.split != "public":
            print(f"[BỎ QUA] {scene.name}: không phải public_set, không có ảnh GT.")
            continue
        renders_dir = renders_root / scene.name / "renders"
        if not renders_dir.exists():
            print(f"[BỎ QUA] {scene.name}: chưa render (thiếu {renders_dir}) — chạy 04_render_test_poses.py trước.")
            continue
        rows = eval_scene(scene, renders_dir, lpips_fn)
        if rows:
            print_stats(scene.name, rows)
            write_csv(renders_root / scene.name / "eval_metrics.csv", rows)
            all_rows.extend(rows)

    if all_rows:
        print_stats("TRUNG BÌNH TOÀN BỘ", all_rows)


if __name__ == "__main__":
    main()
