#!/usr/bin/env python3
"""Tự chấm PSNR/SSIM/LPIPS trên public_set (nơi DUY NHẤT có ảnh GT thật để so sánh).

Không dùng được cho private_set1 (không có ảnh GT) — mục đích của script này là
đánh giá pipeline TRƯỚC khi áp dụng cho private set, đúng lưu ý của BTC ở mục 4.7
đề bài ("kiểm tra kỹ pipeline trên public set trước khi chạy chính thức").

Cách dùng:
    python 05_eval_metrics.py --scene HCM0181
    python 05_eval_metrics.py --all_public
    python 05_eval_metrics.py --all_public --psnr_max 25   # tự chọn PSNR_max khác

Yêu cầu đã chạy 04_render_test_poses.py cho scene đó trước (renders nằm ở
pipeline/work/<scene>/renders/<stem>.png).

Ngoài PSNR/SSIM/LPIPS riêng lẻ, còn tính điểm tổng hợp Score theo ĐÚNG công thức
chính thức của BTC (Đề bài.md mục 8.4):

    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)

QUAN TRỌNG: đề bài KHÔNG công bố giá trị PSNR_max cụ thể (chỉ ghi "ngưỡng được
lựa chọn trước") — --psnr_max ở đây chỉ là GIẢ ĐỊNH để bạn tự ước lượng thứ hạng
tương đối giữa các lần thử nghiệm, KHÔNG phải số chính thức của BTC. Mặc định
30.0 vì đó là mức PSNR "rất tốt" cho cảnh ngoài trời phức tạp theo benchmark
3DGS công khai (Mip-NeRF360 outdoor ~24-27dB, Tanks&Temples ~23dB) — đủ cao để
không bị đạt trần dễ dàng, đủ thấp để không "chôn" kết quả tốt về gần 0. Script
tự in thêm bảng Score ở vài giá trị PSNR_max khác để thấy độ nhạy — nếu có dịp
hỏi BTC được, nên xác nhận lại số thật (xem KE_HOACH_VONG1.md mục "Những điều
cần tự xác nhận thêm").
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


def compute_score(psnr_v: float, ssim_v: float, lpips_v: float, psnr_max: float) -> float:
    """Đúng công thức Đề bài.md mục 8.4. Nếu thiếu LPIPS (chưa cài package), coi
    như 0 cho phần (1 - LPIPS) — KHÔNG đại diện đúng điểm thật, chỉ để không crash."""
    lpips_term = 0.0 if np.isnan(lpips_v) else (1.0 - lpips_v)
    psnr_norm = min(max(psnr_v / psnr_max, 0.0), 1.0)
    return 0.4 * lpips_term + 0.3 * ssim_v + 0.3 * psnr_norm


def write_csv(csv_path: Path, rows: list[tuple], psnr_max: float) -> None:
    """Ghi điểm từng ảnh test ra CSV — dùng để vẽ biểu đồ/so sánh ảnh trong notebook
    (xem cell sau Bước 6 của kaggle_public.ipynb), vì console chỉ in mean/min/max."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "lpips", "score"])
        for stem, psnr_v, ssim_v, lpips_v in rows:
            score = compute_score(psnr_v, ssim_v, lpips_v, psnr_max)
            writer.writerow([stem, psnr_v, ssim_v, lpips_v, score])


def print_stats(name: str, rows: list[tuple], psnr_max: float):
    arr = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f"\n=== {name}: {len(rows)} ảnh ===")
    print(f"  PSNR  mean={arr[:, 0].mean():.3f}  min={arr[:, 0].min():.3f}")
    print(f"  SSIM  mean={arr[:, 1].mean():.4f}  min={arr[:, 1].min():.4f}")
    if not np.isnan(arr[:, 2]).all():
        print(f"  LPIPS mean={arr[:, 2].mean():.4f}  max={arr[:, 2].max():.4f}")

    scores = [compute_score(r[1], r[2], r[3], psnr_max) for r in rows]
    print(f"  Score mean={np.mean(scores):.4f}  min={np.min(scores):.4f}  "
          f"(công thức BTC mục 8.4, PSNR_max={psnr_max} — GIẢ ĐỊNH, xem cảnh báo đầu file)")

    print("  Độ nhạy Score theo PSNR_max khác (để tham khảo, không phải điểm chính thức):")
    for candidate in (20.0, 25.0, 30.0, 35.0, 40.0, 50.0):
        alt_scores = [compute_score(r[1], r[2], r[3], candidate) for r in rows]
        marker = " <- đang dùng" if candidate == psnr_max else ""
        print(f"    PSNR_max={candidate:5.1f} -> Score mean={np.mean(alt_scores):.4f}{marker}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="1 scene public cụ thể, vd HCM0181")
    g.add_argument("--all_public", action="store_true")
    ap.add_argument("--renders_root", default=None, help="Mặc định pipeline/work")
    ap.add_argument("--no_lpips", action="store_true", help="Bỏ qua LPIPS (nếu chưa cài package lpips)")
    ap.add_argument("--psnr_max", type=float, default=30.0,
                     help="Ngưỡng chuẩn hoá PSNR cho công thức Score (mặc định 30.0 — "
                          "GIẢ ĐỊNH, BTC không công bố số thật, xem docstring đầu file)")
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
            print_stats(scene.name, rows, args.psnr_max)
            write_csv(renders_root / scene.name / "eval_metrics.csv", rows, args.psnr_max)
            all_rows.extend(rows)

    if all_rows:
        print_stats("TRUNG BÌNH TOÀN BỘ", all_rows, args.psnr_max)


if __name__ == "__main__":
    main()
