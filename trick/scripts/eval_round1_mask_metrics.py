#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

try:
    import lpips
except ImportError as exc:
    raise SystemExit("Thiếu package lpips. Cài bằng `pip install lpips`.") from exc


def compute_score(psnr_v: float, ssim_v: float, lpips_v: float, psnr_max: float = 50.0) -> float:
    psnr_norm = min(max(psnr_v / psnr_max, 0.0), 1.0)
    return 0.4 * (1.0 - lpips_v) + 0.3 * ssim_v + 0.3 * psnr_norm


def load_rgb01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return arr > 0


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, data_range: float = 1.0) -> float:
    mask3 = mask[..., None]
    denom = mask3.sum() * gt.shape[2]
    if denom <= 0:
        raise ValueError("mask rỗng")
    mse = np.square(gt - pred, dtype=np.float32)[mask3].mean()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range**2) / mse))


def masked_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for c in range(gt.shape[2]):
        ssim_map = structural_similarity(
            gt[:, :, c],
            pred[:, :, c],
            data_range=1.0,
            full=True,
        )[1]
        vals.append(float(ssim_map[mask].mean()))
    return float(np.mean(vals))


def masked_lpips(metric, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    gt_t = (gt_t * 2.0 - 1.0) * mask_t
    pred_t = (pred_t * 2.0 - 1.0) * mask_t
    if torch.cuda.is_available():
        gt_t = gt_t.cuda()
        pred_t = pred_t.cuda()
    with torch.no_grad():
        return float(metric(gt_t, pred_t).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--renders_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_txt", required=True)
    ap.add_argument("--psnr_max", type=float, default=50.0)
    ap.add_argument("--min_coverage", type=float, default=0.001)
    args = ap.parse_args()

    gt_dir = Path(args.dataset_root) / args.scene / "test" / "images"
    renders_dir = Path(args.renders_dir)
    mask_dir = Path(args.mask_dir)
    out_csv = Path(args.out_csv)
    summary_txt = Path(args.summary_txt)

    if not gt_dir.exists():
        raise SystemExit(f"Không thấy GT dir: {gt_dir}")
    if not renders_dir.exists():
        raise SystemExit(f"Không thấy renders dir: {renders_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Không thấy mask dir: {mask_dir}")

    metric = lpips.LPIPS(net="alex")
    if torch.cuda.is_available():
        metric = metric.cuda()

    rows = []
    for gt_path in sorted(gt_dir.glob("*")):
        stem = gt_path.stem
        pred_path = renders_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        if not pred_path.exists() or not mask_path.exists():
            continue

        gt = load_rgb01(gt_path)
        pred = load_rgb01(pred_path)
        mask = load_mask(mask_path)

        if gt.shape != pred.shape:
            raise SystemExit(f"Kích thước GT/pred không khớp ở {stem}")
        if mask.shape != gt.shape[:2]:
            raise SystemExit(f"Kích thước mask không khớp ở {stem}")

        coverage = float(mask.mean())
        if coverage < args.min_coverage:
            print(f"[BỎ QUA] {stem}: coverage={coverage:.6f} < {args.min_coverage}")
            continue

        psnr_v = masked_psnr(gt, pred, mask)
        ssim_v = masked_ssim(gt, pred, mask)
        lpips_v = masked_lpips(metric, gt, pred, mask)
        score_v = compute_score(psnr_v, ssim_v, lpips_v, args.psnr_max)
        rows.append((stem, coverage, psnr_v, ssim_v, lpips_v, score_v))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask_coverage", "psnr", "ssim", "lpips", "score"])
        writer.writerows(rows)

    if not rows:
        raise SystemExit("Không có cặp GT/pred/mask hợp lệ nào để chấm.")

    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=np.float64)
    summary_txt.write_text(
        "\n".join([
            f"Scene: {args.scene}",
            f"Images: {len(rows)}",
            f"Coverage mean: {arr[:,0].mean():.4f}",
            f"Coverage min : {arr[:,0].min():.4f}",
            f"Coverage max : {arr[:,0].max():.4f}",
            f"PSNR mean    : {arr[:,1].mean():.4f}",
            f"SSIM mean    : {arr[:,2].mean():.4f}",
            f"LPIPS mean   : {arr[:,3].mean():.4f}",
            f"Score mean   : {arr[:,4].mean():.4f}",
            f"Saved CSV    : {out_csv}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(summary_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
