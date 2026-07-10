#!/usr/bin/env python3
"""Chẩn đoán: điểm PSNR/SSIM/LPIPS của từng ảnh test có tương quan với khoảng
cách tới camera train GẦN NHẤT hay không — đánh trực tiếp vào giả thuyết (b)
trong Kết quả/Hướng đi.md mục 1 ("pose test nằm ở góc không có ảnh train gần
-> dễ sinh floaters ở vùng khuyết"). Chỉ đọc dữ liệu ĐÃ CÓ SẴN (không train/
render lại gì), dùng để quyết định có nên ưu tiên depth-prior (nếu (b) đúng)
hay hướng khác (nếu không tương quan) trước khi đổ thêm GPU-giờ.

Cách tính khoảng cách: camera center (world space) của từng ảnh test (từ
test_poses.csv, quy ước world->camera giống hệt common/poses.py) so với camera
center của mọi ảnh train (đọc bằng pycolmap.Reconstruction, giống hệt cách
02_validate_frame.py::load_camera_centers đã làm) — lấy khoảng cách NHỎ NHẤT.
test_poses.csv và sparse dùng để train được giả định CHUNG 1 hệ toạ độ (đúng
như toàn bộ pipeline hiện tại đang giả định — không tự align Sim3 ở đây).

Yêu cầu đã chạy trước (cho 1 scene public):
    python 01_run_colmap.py --scene <scene>       (tạo work/<scene>/colmap/dense/sparse/0)
    python 04_render_test_poses.py --scene <scene>
    python 05_eval_metrics.py --scene <scene>      (tạo work/<scene>/eval_metrics.csv)

Cách dùng:
    python 09_diagnose_distance.py --scene hcm0031

Output: in ra tương quan Pearson + so sánh nhóm ảnh GẦN vs XA camera train
nhất, và ghi chi tiết từng ảnh ra work/<scene>/diagnose_distance.csv (dùng để
tự vẽ scatter plot trong notebook nếu muốn).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import read_test_poses, qvec2rotmat


def load_train_camera_centers(sparse_dir: Path) -> dict:
    """image_name -> camera center (3,) world space — y hệt cách
    02_validate_frame.py::load_camera_centers đã đối chiếu source pycolmap."""
    rec = pycolmap.Reconstruction(sparse_dir)
    return {image.name: np.array(image.projection_center()) for _, image in rec.images.items()}


def test_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """C_world = -R_wc^T @ t_wc — đúng quy ước world->camera COLMAP (xem
    docstring đầu common/poses.py)."""
    R_wc = qvec2rotmat(qvec)
    return -R_wc.T @ tvec


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Scene public (cần đã có eval_metrics.csv)")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    work_dir = pipeline_root / "work" / scene.name
    sparse_dir = work_dir / "colmap" / "dense" / "sparse" / "0"
    eval_csv = work_dir / "eval_metrics.csv"

    if not sparse_dir.exists():
        raise SystemExit(f"Không thấy {sparse_dir} — chạy 01_run_colmap.py --scene {scene.name} trước.")
    if not eval_csv.exists():
        raise SystemExit(
            f"Không thấy {eval_csv} — chạy 04_render_test_poses.py rồi 05_eval_metrics.py "
            f"--scene {scene.name} trước."
        )

    print(f"Đang đọc camera train từ {sparse_dir} ...")
    train_centers = load_train_camera_centers(sparse_dir)
    if not train_centers:
        raise SystemExit(f"{sparse_dir}: không có camera nào trong sparse — kiểm tra lại dữ liệu.")
    train_pts = np.array(list(train_centers.values()))
    print(f"  -> {len(train_centers)} camera train.")

    test_poses = {p.image_name: p for p in read_test_poses(scene.test_poses_csv)}

    rows: list[tuple[str, float, float, float, float]] = []
    with open(eval_csv, newline="") as f:
        for r in csv.DictReader(f):
            stem = r["image"]
            # eval_metrics.csv ghi theo stem (không đuôi file), test_poses.csv giữ
            # nguyên image_name gốc (có thể .JPG) — khớp lại theo stem.
            match = next((name for name in test_poses if Path(name).stem == stem), None)
            if match is None:
                print(f"  [BỎ QUA] {stem}: không khớp được với test_poses.csv")
                continue
            pose = test_poses[match]
            C = test_camera_center(pose.qvec, pose.tvec)
            dist = float(np.min(np.linalg.norm(train_pts - C, axis=1)))
            rows.append((stem, dist, float(r["psnr"]), float(r["ssim"]), float(r["lpips"])))

    if len(rows) < 5:
        raise SystemExit(f"Chỉ khớp được {len(rows)} ảnh — không đủ để phân tích (cần >= 5).")

    dists = np.array([row[1] for row in rows])
    psnrs = np.array([row[2] for row in rows])
    ssims = np.array([row[3] for row in rows])
    lpipss = np.array([row[4] for row in rows])

    scene_diag = float(np.linalg.norm(train_pts.max(axis=0) - train_pts.min(axis=0)))
    print(f"\nĐường kính cảnh (bounding box camera train, dùng làm thang tham chiếu): {scene_diag:.3f}")
    print(f"Khoảng cách tới camera train gần nhất — {len(rows)} ảnh test:")
    print(f"  mean={dists.mean():.3f}  median={np.median(dists):.3f}  min={dists.min():.3f}  max={dists.max():.3f}")
    print(f"  (theo % đường kính cảnh: mean={100 * dists.mean() / scene_diag:.2f}%)")

    print("\nTương quan Pearson (khoảng cách vs từng chỉ số — |r| càng gần 1 càng tương quan mạnh):")
    print(f"  distance vs PSNR : r={pearson_r(dists, psnrs):+.3f}  "
          f"(kỳ vọng ÂM nếu giả thuyết (b) đúng — xa hơn thì PSNR thấp hơn)")
    print(f"  distance vs SSIM : r={pearson_r(dists, ssims):+.3f}  (kỳ vọng ÂM)")
    print(f"  distance vs LPIPS: r={pearson_r(dists, lpipss):+.3f}  (kỳ vọng DƯƠNG — xa hơn thì LPIPS cao hơn = tệ hơn)")

    order = np.argsort(dists)
    half = len(order) // 2
    near_idx, far_idx = order[:half], order[-half:]

    def group_stats(idx: np.ndarray, name: str) -> None:
        print(f"  {name:9s} (n={len(idx):2d}, dist mean={dists[idx].mean():.3f}): "
              f"PSNR={psnrs[idx].mean():.3f}  SSIM={ssims[idx].mean():.4f}  LPIPS={lpipss[idx].mean():.4f}")

    print("\nSo sánh nửa GẦN camera train nhất vs nửa XA nhất:")
    group_stats(near_idx, "Gần nhất")
    group_stats(far_idx, "Xa nhất")

    gap_psnr = psnrs[near_idx].mean() - psnrs[far_idx].mean()
    gap_lpips = lpipss[far_idx].mean() - lpipss[near_idx].mean()
    print(f"\nChênh lệch PSNR (gần - xa) = {gap_psnr:+.3f} dB, chênh lệch LPIPS (xa - gần) = {gap_lpips:+.4f}")
    if gap_psnr > 1.0 or gap_lpips > 0.02:
        print("=> CÓ dấu hiệu tương quan rõ: ảnh test xa camera train render tệ hơn đáng kể — "
              "giả thuyết (b) floaters/thiếu ảnh train gần NHIỀU KHẢ NĂNG đúng ở scene này, "
              "nên ưu tiên đẩy mạnh depth-prior.")
    else:
        print("=> KHÔNG thấy tương quan rõ giữa khoảng cách và chất lượng ở scene này — giả "
              "thuyết (b) có thể KHÔNG phải nguyên nhân chính, cân nhắc hướng (c)/VRAM-Gaussian "
              "budget thay vì dồn thêm lực vào depth-prior.")

    out_csv = work_dir / "diagnose_distance.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "dist_to_nearest_train_cam", "psnr", "ssim", "lpips"])
        writer.writerows(rows)
    print(f"\nĐã ghi chi tiết từng ảnh ra {out_csv} (dùng để tự vẽ scatter plot nếu muốn).")


if __name__ == "__main__":
    main()
