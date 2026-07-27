#!/usr/bin/env python3
import argparse
import csv
import json
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


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
    ], dtype=np.float64)


def read_test_poses(csv_path: Path) -> dict:
    poses = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stem = Path(r["image_name"]).stem
            poses[stem] = {
                "qvec": np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                "tvec": np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                "fx": float(r["fx"]),
                "fy": float(r["fy"]),
                "cx": float(r["cx"]),
                "cy": float(r["cy"]),
                "width": int(float(r["width"])),
                "height": int(float(r["height"])),
            }
    return poses


def project_points(points_world: np.ndarray, pose: dict) -> tuple[np.ndarray, np.ndarray]:
    rot = qvec2rotmat(pose["qvec"])
    cam = (rot @ points_world.T).T + pose["tvec"]
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(points_world), 2), dtype=np.float64)
    uv[valid, 0] = pose["fx"] * (cam[valid, 0] / z[valid]) + pose["cx"]
    uv[valid, 1] = pose["fy"] * (cam[valid, 1] / z[valid]) + pose["cy"]
    return uv, valid


def tower_crop_box(pose: dict, corners: np.ndarray) -> tuple[int, int, int, int] | None:
    uv, valid = project_points(corners, pose)
    valid_uv = uv[valid]
    if len(valid_uv) < 3:
        return None
    x0 = max(int(np.floor(valid_uv[:, 0].min())), 0)
    y0 = max(int(np.floor(valid_uv[:, 1].min())), 0)
    x1 = min(int(np.ceil(valid_uv[:, 0].max())), pose["width"])
    y1 = min(int(np.ceil(valid_uv[:, 1].max())), pose["height"])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def skyline_crop_box(width: int, height: int, top_frac: float) -> tuple[int, int, int, int]:
    y1 = max(int(round(height * top_frac)), 1)
    return 0, 0, width, y1


def compute_metrics(gt: np.ndarray, pred: np.ndarray, metric, psnr_max: float) -> tuple[float, float, float, float]:
    psnr_v = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_v = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
    t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    if torch.cuda.is_available():
        t_gt = t_gt.cuda()
        t_pred = t_pred.cuda()
    with torch.no_grad():
        lpips_v = float(metric(t_gt, t_pred).item())
    score = compute_score(psnr_v, ssim_v, lpips_v, psnr_max)
    return psnr_v, ssim_v, lpips_v, score


def write_region_csv(out_csv: Path, rows: list) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim", "lpips", "score", "x0", "y0", "x1", "y1"])
        writer.writerows(rows)
    if rows:
        arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
        print(f"  [{out_csv.name}] n={len(rows)} PSNR={arr[:,0].mean():.4f} SSIM={arr[:,1].mean():.4f} "
              f"LPIPS={arr[:,2].mean():.4f} Score={arr[:,3].mean():.4f}")
    else:
        print(f"  [{out_csv.name}] KHÔNG có ảnh nào tính được crop hợp lệ (0 dòng).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--renders_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--psnr_max", type=float, default=50.0)
    ap.add_argument("--skyline_top_frac", type=float, default=None)
    ap.add_argument("--tower_bbox3d_json", default=None)
    args = ap.parse_args()

    gt_dir = Path(args.dataset_root) / args.scene / "test" / "images"
    renders_dir = Path(args.renders_dir)
    out_csv = Path(args.out_csv)
    if not gt_dir.exists():
      raise SystemExit(f"Không thấy GT dir: {gt_dir}")
    if not renders_dir.exists():
      raise SystemExit(f"Không thấy renders dir: {renders_dir}")

    test_poses = None
    tower_corners = None
    if args.tower_bbox3d_json or args.skyline_top_frac is not None:
        poses_csv = Path(args.dataset_root) / args.scene / "test" / "test_poses.csv"
        if poses_csv.exists():
            test_poses = read_test_poses(poses_csv)
        else:
            print(f"[CẢNH BÁO] Không thấy {poses_csv}, bỏ qua region crop.")

    if args.tower_bbox3d_json:
        bbox_path = Path(args.tower_bbox3d_json)
        if bbox_path.exists():
            tower_corners = np.asarray(json.loads(bbox_path.read_text(encoding="utf-8"))["corners"], dtype=np.float64)
        else:
            print(f"[CẢNH BÁO] Không thấy {bbox_path}, bỏ qua tower-crop.")

    metric = lpips.LPIPS(net="alex")
    if torch.cuda.is_available():
        metric = metric.cuda()

    rows = []
    tower_rows = []
    skyline_rows = []
    gt_paths = sorted(gt_dir.glob("*"))
    for gt_path in gt_paths:
        stem = gt_path.stem
        pred_path = renders_dir / f"{stem}.png"
        if not pred_path.exists():
            print(f"[THIẾU] {pred_path.name}")
            continue

        gt = load_img01(gt_path)
        pred = load_img01(pred_path)
        psnr_v, ssim_v, lpips_v, score = compute_metrics(gt, pred, metric, args.psnr_max)
        rows.append((stem, psnr_v, ssim_v, lpips_v, score))

        pose = test_poses.get(stem) if test_poses is not None else None

        if tower_corners is not None and pose is not None:
            box = tower_crop_box(pose, tower_corners)
            if box is not None:
                x0, y0, x1, y1 = box
                pm, sm, lm, scm = compute_metrics(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], metric, args.psnr_max)
                tower_rows.append((stem, pm, sm, lm, scm, x0, y0, x1, y1))

        if args.skyline_top_frac is not None and pose is not None:
            x0, y0, x1, y1 = skyline_crop_box(pose["width"], pose["height"], args.skyline_top_frac)
            pm, sm, lm, scm = compute_metrics(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1], metric, args.psnr_max)
            skyline_rows.append((stem, pm, sm, lm, scm, x0, y0, x1, y1))

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

    if tower_corners is not None:
        write_region_csv(out_csv.with_name(out_csv.stem + "_tower_crop.csv"), tower_rows)
    if args.skyline_top_frac is not None:
        write_region_csv(out_csv.with_name(out_csv.stem + "_skyline_crop.csv"), skyline_rows)


if __name__ == "__main__":
    main()
