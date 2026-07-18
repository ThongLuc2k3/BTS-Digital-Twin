#!/usr/bin/env python3
"""Xác định vùng 3D của ăn-ten (hoặc bất kỳ vùng nhỏ nào muốn tập trung train hơn)
từ 1 ảnh tham chiếu + 1 khung pixel do người dùng chỉ ra, rồi chiếu vùng đó vào
TẤT CẢ ảnh train để ra 1 file JSON dùng chung cho cả 2 ý tưởng:

  - Trọng số hoá loss theo vùng (loss masking) — vùng ăn-ten được nhân loss cao hơn.
  - Chọn lại tần suất camera (view resampling) — ảnh thấy ăn-ten rõ được train.py
    chọn thường xuyên hơn ảnh toàn cảnh.

CHỈ dùng cho 5 scene BTS (domain="bts") — bonsai/chair không có ăn-ten, không áp
dụng kỹ thuật này cho 2 scene đó (xem plan.md mục 9 "việc không được làm").

Cách hoạt động: sparse COLMAP (dense/sparse/0, do 01_run_colmap.py tạo) lưu sẵn,
với MỖI ảnh, danh sách keypoint 2D và (nếu có) điểm 3D tương ứng đã tam giác hoá.
Chỉ cần 1 khung pixel bao quanh ăn-ten trên 1 ảnh, lọc ra các keypoint của ảnh đó
rơi vào khung, lấy điểm 3D tương ứng — không cần đoán toạ độ 3D bằng tay, không
cần chạy model segmentation riêng.

Cách dùng:
    python 07_build_antenna_weights.py --scene HCM0421 --ref_image DJI_0123.JPG \
        --box 812 140 960 640

    (box: x_min y_min x_max y_max theo pixel, gốc (0,0) ở góc trên-trái — mở ảnh
    pipeline/work/<scene>/colmap/dense/images/<ref_image> hoặc ảnh train gốc bằng
    trình xem ảnh bất kỳ để đọc toạ độ khung bao quanh ăn-ten.)

Output: pipeline/work/<scene>/antenna_weights.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene


def _detect_points2d_scale(rec: "pycolmap.Reconstruction") -> float:
    """BUG THẬT tìm thấy + fix (2026-07-18) — giống hệt bug đã phát hiện ở
    `11_trr_refine.py::_detect_points2d_scale`, nhưng đây là sparse KHÁC
    (`pipeline/work/<scene>/colmap/dense/sparse/0`, do `01_run_colmap.py` tự sinh qua
    COLMAP `image_undistorter`, không phải `scene.provided_sparse_dir` gốc BTC cấp).
    Verify thực nghiệm: `image_undistorter` viết lại ĐÚNG `camera.width/height` khớp
    ảnh `train/images/` thật, nhưng KHÔNG viết lại `points2D[i].xy` — field này vẫn ở
    ĐỘ PHÂN GIẢI GỐC (trước khi BTC hạ xuống), lệch `camera.width/height/fx/fy` đúng hệ
    số "Scale" ở plan.md mục 2. Đo trực tiếp bằng dữ liệu COLMAP thật cục bộ (không suy
    đoán): HCM0421 lệch 3.904x, chair 1.500x, bonsai 1.000x (không lệch) — khớp chính
    xác plan.md. Nếu KHÔNG sửa, `find_antenna_bbox3d()` so khung pixel người dùng nhập
    (theo đúng kích thước ảnh thật họ đang xem) với `point2d.xy` (lớn hơn ~4x cho
    HCM0421) — gần như chắc chắn lọc ra 0 điểm (báo lỗi rõ) hoặc tệ hơn, tình cờ lọc
    trúng vài điểm SAI hoàn toàn vị trí, ra bbox 3D rác mà không có lỗi báo."""
    best_img = max(rec.images.values(), key=lambda im: im.num_points3D)
    obs = best_img.get_observation_points2D()
    ratios = []
    for o in obs:
        proj = best_img.project_point(rec.points3D[o.point3D_id].xyz)
        if proj is None or proj[0] <= 1 or proj[1] <= 1:
            continue
        ratios.append(o.xy[0] / proj[0])
        ratios.append(o.xy[1] / proj[1])
    if not ratios:
        return 1.0
    scale = float(np.median(ratios))
    if abs(scale - 1.0) > 0.02:
        print(f"  [LƯU Ý] points2D.xy lệch camera intrinsics theo hệ số {scale:.4f}x "
              f"(khớp cột 'Scale' của plan.md) — tự chia lại toạ độ trước khi lọc khung.")
    return scale


def find_antenna_bbox3d(rec: "pycolmap.Reconstruction", ref_image_name: str,
                         box: tuple[float, float, float, float], margin: float,
                         points2d_scale: float) -> np.ndarray:
    ref_image = None
    for image in rec.images.values():
        if image.name == ref_image_name:
            ref_image = image
            break
    if ref_image is None:
        available = sorted(image.name for image in rec.images.values())
        raise SystemExit(
            f"Không tìm thấy ảnh '{ref_image_name}' trong sparse — vài tên có sẵn: {available[:10]}..."
        )

    x0, y0, x1, y1 = box
    pts3d = []
    for point2d in ref_image.points2D:
        if not point2d.has_point3D():
            continue
        x, y = point2d.xy
        x, y = x / points2d_scale, y / points2d_scale  # về đúng hệ toạ độ ảnh thật (--box nhập theo)
        if x0 <= x <= x1 and y0 <= y <= y1:
            pts3d.append(rec.points3D[point2d.point3D_id].xyz)

    if len(pts3d) < 3:
        raise SystemExit(
            f"Chỉ tìm thấy {len(pts3d)} điểm 3D trong khung {box} của ảnh '{ref_image_name}' — "
            f"quá ít để tin cậy (cần >= 3). Thử mở rộng khung hoặc chọn ảnh khác thấy rõ ăn-ten hơn."
        )

    pts3d = np.array(pts3d)
    bbox_min = pts3d.min(axis=0)
    bbox_max = pts3d.max(axis=0)
    extent = bbox_max - bbox_min
    bbox_min = bbox_min - extent * margin
    bbox_max = bbox_max + extent * margin
    print(f"-> {len(pts3d)} điểm 3D trong khung tham chiếu. Bbox 3D (đã nới {margin:.0%}): "
          f"min={bbox_min.tolist()} max={bbox_max.tolist()}")
    return np.stack([bbox_min, bbox_max])


def project_bbox_to_images(rec: "pycolmap.Reconstruction", bbox3d: np.ndarray) -> dict:
    bbox_min, bbox_max = bbox3d
    corners = np.array([
        [x, y, z]
        for x in (bbox_min[0], bbox_max[0])
        for y in (bbox_min[1], bbox_max[1])
        for z in (bbox_min[2], bbox_max[2])
    ])

    images_out = {}
    for image in rec.images.values():
        if not image.has_pose:
            continue
        camera = image.camera
        if camera is None:
            continue
        cam_from_world = image.cam_from_world()

        pixels = []
        for corner in corners:
            cam_point = cam_from_world * corner
            if cam_point[2] <= 0:
                continue  # sau lưng camera, bỏ qua
            image_point = camera.img_from_cam(cam_point)
            if image_point is not None:
                pixels.append(image_point)

        if not pixels:
            continue
        pixels = np.array(pixels)
        x0 = float(np.clip(pixels[:, 0].min(), 0, camera.width))
        x1 = float(np.clip(pixels[:, 0].max(), 0, camera.width))
        y0 = float(np.clip(pixels[:, 1].min(), 0, camera.height))
        y1 = float(np.clip(pixels[:, 1].max(), 0, camera.height))
        if x1 <= x0 or y1 <= y0:
            continue  # hộp chiếu ra ngoài khung hình hoàn toàn

        coverage = ((x1 - x0) * (y1 - y0)) / (camera.width * camera.height)
        images_out[image.name] = {"box": [x0, y0, x1, y1], "coverage": coverage}

    return images_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ref_image", required=True, help="Tên file ảnh (đúng tên trong train/images/) thấy rõ ăn-ten")
    ap.add_argument("--box", nargs=4, type=float, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                     help="Khung pixel bao quanh ăn-ten trên ảnh --ref_image")
    ap.add_argument("--margin", type=float, default=0.15, help="Nới rộng bbox 3D thêm bao nhiêu %% mỗi chiều (mặc định 15%%)")
    ap.add_argument("--weight", type=float, default=4.0, help="Hệ số nhân loss trong vùng ăn-ten lúc train (mặc định 4.0)")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    if scene.domain != "bts":
        raise SystemExit(
            f"{scene.name}: domain='{scene.domain}', không phải scene BTS — antenna-focus chỉ áp dụng "
            f"cho 5 scene BTS (HCM04xx/05xx/06xx), không dùng cho bonsai/chair."
        )
    pipeline_root = Path(__file__).resolve().parents[1]
    sparse_dir = pipeline_root / "work" / scene.name / "colmap" / "dense" / "sparse" / "0"
    if not sparse_dir.exists():
        raise SystemExit(f"Không thấy {sparse_dir} — chạy 01_run_colmap.py --scene {scene.name} trước.")

    rec = pycolmap.Reconstruction(sparse_dir)
    points2d_scale = _detect_points2d_scale(rec)
    bbox3d = find_antenna_bbox3d(rec, args.ref_image, tuple(args.box), args.margin, points2d_scale)
    images_out = project_bbox_to_images(rec, bbox3d)

    n_total = rec.num_reg_images()
    print(f"-> Chiếu bbox vào {len(images_out)}/{n_total} ảnh train (ảnh còn lại: hộp nằm ngoài khung hình hoặc sau lưng camera).")

    out_path = pipeline_root / "work" / scene.name / "antenna_weights.json"
    out_path.write_text(json.dumps({
        "scene": scene.name,
        "ref_image": args.ref_image,
        "box_input": list(args.box),
        "margin": args.margin,
        "weight_value": args.weight,
        "bbox_3d": bbox3d.tolist(),
        "images": images_out,
    }, indent=2), encoding="utf-8")
    print(f"-> Đã ghi {out_path}")


if __name__ == "__main__":
    main()
