#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
except ImportError as exc:
    raise SystemExit("Thiếu package lpips. Cài bằng `pip install lpips`.") from exc


def load_img01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def compute_score(psnr_v: float, ssim_v: float, lpips_v: float, psnr_max: float = 50.0) -> float:
    psnr_norm = min(max(psnr_v / psnr_max, 0.0), 1.0)
    return 0.4 * (1.0 - lpips_v) + 0.3 * ssim_v + 0.3 * psnr_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--renders_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--psnr_max", type=float, default=50.0)
    args = ap.parse_args()

    gt_dir = Path(args.dataset_root) / args.scene / "test" / "images"
    renders_dir = Path(args.renders_dir)
    out_csv = Path(args.out_csv)
    if not gt_dir.exists():
      raise SystemExit(f"Không thấy GT dir: {gt_dir}")
    if not renders_dir.exists():
      raise SystemExit(f"Không thấy renders dir: {renders_dir}")

    metric = lpips.LPIPS(net="alex")
    if torch.cuda.is_available():
        metric = metric.cuda()

    rows = []
    gt_paths = sorted(gt_dir.glob("*"))
    for gt_path in gt_paths:
        stem = gt_path.stem
        pred_path = renders_dir / f"{stem}.png"
        if not pred_path.exists():
            print(f"[THIẾU] {pred_path.name}")
            continue

        gt = load_img01(gt_path)
        pred = load_img01(pred_path)
        psnr_v = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_v = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)

        t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        if torch.cuda.is_available():
            t_gt = t_gt.cuda()
            t_pred = t_pred.cuda()
        with torch.no_grad():
            lpips_v = float(metric(t_gt, t_pred).item())
        score = compute_score(psnr_v, ssim_v, lpips_v, args.psnr_max)
        rows.append((stem, psnr_v, ssim_v, lpips_v, score))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "lpips", "score"])
        writer.writerows(rows)

    if not rows:
        raise SystemExit("Không có cặp GT/pred nào để chấm.")

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    print(f"Scene: {args.scene}")
    print(f"Images: {len(rows)}")
    print(f"PSNR mean : {arr[:,0].mean():.4f}")
    print(f"SSIM mean : {arr[:,1].mean():.4f}")
    print(f"LPIPS mean: {arr[:,2].mean():.4f}")
    print(f"Score mean: {arr[:,3].mean():.4f}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
