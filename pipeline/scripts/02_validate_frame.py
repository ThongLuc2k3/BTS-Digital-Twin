#!/usr/bin/env python3
"""Sanity-check hệ toạ độ (TUỲ CHỌN — không bắt buộc).

Dataset có sparse hợp lệ ở cả 13/13 scene, nên pipeline mặc định dùng THẲNG
sparse có sẵn (xem `01_run_colmap.py` và
`common/colmap_runner.py::use_provided_sparse`), không cần tự dựng lại COLMAP.
Script này vẫn giữ lại, hữu ích khi:
- Muốn đối chiếu/kiểm tra thêm cho chắc trước khi tin tưởng hoàn toàn.
- Nghi ngờ chất lượng sparse của 1 scene cụ thể nào đó.

Script này: tự chạy COLMAP trên train/images/ của HCM0249, rồi so sánh camera
centers (projection_center) của các ảnh train với camera centers có sẵn trong
sparse gốc BTC cung cấp — CÙNG một tập ảnh, nên đây là phép so sánh trực tiếp,
không suy đoán.

Đọc kết quả:
- "RAW residual" nhỏ (~0, tính theo % đường kính cảnh)  => 2 hệ trùng khớp tự nhiên.
- "RAW residual" lớn nhưng "ALIGNED residual" (sau Sim3) nhỏ => 2 SfM đều hợp lệ
  nhưng khác gauge (chỉ còn liên quan nếu dùng --force_own_colmap cho scene nào đó).
- Cả 2 residual đều lớn => có vấn đề khác (COLMAP thất bại một phần, ảnh thiếu, ...).
"""
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.colmap_runner import run_colmap_scene
from common.alignment import umeyama_alignment, raw_residuals
from common.poses import representative_intrinsics

SCENE_NAME = "HCM0249"  # scene DUY NHẤT có sparse gốc hợp lệ, xem Dataset/README.md


def load_camera_centers(rec: pycolmap.Reconstruction) -> dict:
    """image_name -> camera center (3,) trong world space của reconstruction đó."""
    centers = {}
    for _, image in rec.images.items():
        centers[image.name] = np.array(image.projection_center())
    return centers


def scene_scale(centers: dict) -> float:
    pts = np.array(list(centers.values()))
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def main():
    scene = get_scene(SCENE_NAME)
    if not scene.has_valid_provided_sparse():
        raise SystemExit(
            f"{SCENE_NAME}: sparse gốc không hợp lệ/không tồn tại — không thể chạy "
            f"kiểm định này. Kiểm tra lại dataset đã tải/giải nén đầy đủ chưa "
            f"(xem BTS_DATASET_ROOT có trỏ đúng chỗ không, và __MACOSX/ có bị lẫn vào không)."
        )

    print(f"Đang load sparse GỐC (BTC cung cấp) từ {scene.provided_sparse_dir} ...")
    official_rec = pycolmap.Reconstruction(scene.provided_sparse_dir)
    official_centers = load_camera_centers(official_rec)
    print(f"  -> {len(official_centers)} ảnh có pose trong sparse gốc.")

    print(f"\nTự chạy COLMAP trên train/images/ của {SCENE_NAME} (không dùng sparse gốc) ...")
    fx, cx, cy, width, height = representative_intrinsics(scene.test_poses_csv)
    work_root = Path(__file__).resolve().parents[1] / "work" / scene.name
    workdir = work_root / "colmap_own_validation"
    log_path = work_root / "02_validate_frame.log"
    result = run_colmap_scene(
        images_dir=scene.train_images_dir,
        workdir=workdir,
        matching="sequential",
        camera_model="SIMPLE_RADIAL",
        camera_params_prior=f"{fx},{cx},{cy},0.0",
        log_path=log_path,
    )
    own_rec = pycolmap.Reconstruction(result["sparse_dir"])
    own_centers = load_camera_centers(own_rec)
    print(f"  -> {len(own_centers)} ảnh có pose trong sparse tự chạy. (log chi tiết: {result['log_path']})")

    common_names = sorted(set(official_centers) & set(own_centers))
    print(f"\nSố ảnh xuất hiện ở CẢ 2 reconstruction: {len(common_names)}")
    if len(common_names) < 3:
        raise SystemExit(
            "Không đủ ảnh chung (<3) để ước lượng Sim3 — COLMAP tự chạy có thể đã "
            "thất bại một phần. Kiểm tra log ở bước mapping (--matching exhaustive)."
        )

    src = np.array([own_centers[n] for n in common_names])       # tự chạy
    dst = np.array([official_centers[n] for n in common_names])  # gốc BTC

    scale_ref = scene_scale(dst)
    raw = raw_residuals(src, dst)
    s, R, t, aligned_res, _ = umeyama_alignment(src, dst)

    def stats(x):
        return f"mean={x.mean():.4f}  median={np.median(x):.4f}  max={x.max():.4f}"

    print("\n" + "=" * 70)
    print(f"Đường kính cảnh (dùng làm thang tham chiếu): {scale_ref:.4f}")
    print(f"RAW residual (không align)      : {stats(raw)}   ({100*raw.mean()/scale_ref:.2f}% đường kính cảnh)")
    print(f"ALIGNED residual (sau Sim3)      : {stats(aligned_res)}   ({100*aligned_res.mean()/scale_ref:.2f}% đường kính cảnh)")
    print(f"Sim3 ước lượng được: scale={s:.6f}")
    print(f"  (scale gần 1.0 và residual RAW đã nhỏ sẵn => 2 hệ trùng khớp tự nhiên, tốt nhất)")
    print("=" * 70)

    if raw.mean() / scale_ref < 0.02:
        print("\n=> KẾT LUẬN: RAW residual rất nhỏ. Hệ toạ độ tự chạy COLMAP TRÙNG KHỚP tự nhiên "
              "với hệ của BTC. AN TOÀN dùng COLMAP tự chạy cho các scene còn lại và tin tưởng "
              "trực tiếp test_poses.csv (không cần align gì thêm).")
    elif aligned_res.mean() / scale_ref < 0.02:
        print("\n=> KẾT LUẬN: 2 hệ toạ độ khác gauge (RAW residual lớn) nhưng nội tại nhất quán "
              "(ALIGNED residual nhỏ). RỦI RO CAO cho 7 scene private còn lại KHÔNG có sparse gốc "
              "để đối chiếu — không có cách suy ra đúng Sim3 cho chúng. BÁO BTC NGAY và tham khảo "
              "mục 4 trong KE_HOACH_VONG1.md.")
    else:
        print("\n=> KẾT LUẬN: Cả 2 residual đều lớn — có vấn đề khác (COLMAP thất bại một phần, "
              "ảnh không đủ liên kết, sai camera model...). Kiểm tra lại log mapping phía trên, "
              "thử --matching exhaustive, hoặc kiểm tra tỉ lệ ảnh đăng ký được.")


if __name__ == "__main__":
    main()
