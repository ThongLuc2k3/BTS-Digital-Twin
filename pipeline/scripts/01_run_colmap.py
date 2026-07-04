#!/usr/bin/env python3
"""Chạy COLMAP cho 1 scene (hoặc toàn bộ scene) bằng pycolmap.

Ví dụ:
    python 01_run_colmap.py --scene HCM0181
    python 01_run_colmap.py --scene HCM0249 --matching exhaustive
    python 01_run_colmap.py --all --split public

Output: pipeline/work/<scene>/colmap/dense/{images/, sparse/0/}
        -> đây chính là thư mục để đưa vào train.py của gaussian-splatting.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import Scene, all_scenes, get_scene
from common.colmap_runner import run_colmap_scene
from common.poses import representative_intrinsics

WORK_ROOT = Path(__file__).resolve().parents[1] / "work"


def process_scene(scene: Scene, matching: str, camera_model: str, use_prior: bool, overwrite: bool):
    print(f"\n===== Scene {scene.name} ({scene.split}) =====")
    workdir = WORK_ROOT / scene.name / "colmap"

    camera_params_prior = None
    if use_prior and scene.test_poses_csv.exists():
        fx, cx, cy, width, height = representative_intrinsics(scene.test_poses_csv)
        if camera_model == "SIMPLE_RADIAL":
            camera_params_prior = f"{fx},{cx},{cy},0.0"
        elif camera_model == "PINHOLE":
            camera_params_prior = f"{fx},{fx},{cx},{cy}"
        print(f"Dùng prior nội tham số từ test_poses.csv: {camera_params_prior} "
              f"(giả định train/test cùng 1 camera vật lý trong 1 chuyến bay)")

    result = run_colmap_scene(
        images_dir=scene.train_images_dir,
        workdir=workdir,
        matching=matching,
        camera_model=camera_model,
        camera_params_prior=camera_params_prior,
        overwrite=overwrite,
    )
    n_total = len(list(scene.train_images_dir.glob("*")))
    print(f"-> Đăng ký được {result['num_reg_images']}/{n_total} ảnh, "
          f"{result['num_points3D']} điểm 3D thưa.")
    if result["num_reg_images"] < 0.8 * n_total:
        print(f"[CẢNH BÁO] Tỉ lệ đăng ký ảnh thấp (<80%) — cân nhắc thử "
              f"--matching exhaustive hoặc kiểm tra lại ảnh scene {scene.name}.")
    print(f"Dense (dùng cho train 3DGS): {result['dense_dir']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="Tên 1 scene, vd HCM0181, HCM0249, hcm0031")
    g.add_argument("--all", action="store_true", help="Chạy toàn bộ scene")
    ap.add_argument("--split", choices=["public", "private"], default=None,
                     help="Chỉ dùng với --all, giới hạn public_set hoặc private_set1")
    ap.add_argument("--matching", default="sequential", choices=["sequential", "exhaustive"])
    ap.add_argument("--camera_model", default="SIMPLE_RADIAL",
                     help="SIMPLE_RADIAL (khớp dữ liệu gốc, có méo nhẹ) hoặc PINHOLE")
    ap.add_argument("--no_prior", action="store_true",
                     help="Không dùng fx/cx/cy từ test_poses.csv làm prior, để COLMAP tự đoán từ EXIF/heuristic")
    ap.add_argument("--overwrite", action="store_true", help="Chạy lại từ đầu, xoá database.db cũ")
    args = ap.parse_args()

    if args.scene:
        scenes = [get_scene(args.scene)]
    else:
        scenes = all_scenes(args.split)

    for scene in scenes:
        process_scene(scene, args.matching, args.camera_model, not args.no_prior, args.overwrite)


if __name__ == "__main__":
    main()
