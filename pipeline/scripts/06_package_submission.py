#!/usr/bin/env python3
"""Đóng gói submission_round1.zip từ pipeline/work/<scene>/renders/ cho 8 scene
private_set1, kèm kiểm tra tự động TRƯỚC khi nén (đủ scene/ảnh/đúng kích thước) —
đúng checklist ở KE_HOACH_VONG1.md mục 7.

Cách dùng:
    python 06_package_submission.py --out submission_round1.zip
    python 06_package_submission.py --check_only submission_round1.zip   # chỉ kiểm tra zip có sẵn

Về tên file trong zip (xem KE_HOACH_VONG1.md mục 4, câu hỏi #3 — CHƯA có xác nhận
từ BTC, mặc định dùng phương án literal vì đó là câu chữ trong đề bài):
    --filename_mode literal   (mặc định): giữ NGUYÊN chuỗi image_name trong CSV
                                           (kể cả đuôi .JPG gốc), nội dung file vẫn
                                           là PNG thật (chỉ đổi tên, không đổi định dạng).
    --filename_mode png_ext             : đổi đuôi thành .png, theo đúng ví dụ minh
                                           hoạ "0001.png" trong đề bài.
"""
import argparse
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import all_scenes, Scene
from common.poses import read_test_poses


def target_filename(image_name: str, mode: str) -> str:
    if mode == "literal":
        return image_name
    if mode == "png_ext":
        return f"{Path(image_name).stem}.png"
    raise ValueError(mode)


def check_scene(scene: Scene, renders_dir: Path) -> list[str]:
    errors = []
    poses = read_test_poses(scene.test_poses_csv)
    for pose in poses:
        stem = Path(pose.image_name).stem
        render_path = renders_dir / f"{stem}.png"
        if not render_path.exists():
            errors.append(f"{scene.name}/{pose.image_name}: THIẾU render ({render_path})")
            continue
        with Image.open(render_path) as im:
            if im.size != (pose.width, pose.height):
                errors.append(
                    f"{scene.name}/{pose.image_name}: kích thước {im.size} != "
                    f"yêu cầu ({pose.width},{pose.height})"
                )
    n_found = len(list(renders_dir.glob("*.png")))
    if n_found != len(poses):
        errors.append(f"{scene.name}: số file render ({n_found}) != số pose yêu cầu ({len(poses)})")
    return errors


def build_zip(out_path: Path, renders_root: Path, filename_mode: str, scenes: list[Scene]):
    all_errors = []
    for scene in scenes:
        renders_dir = renders_root / scene.name / "renders"
        if not renders_dir.exists():
            all_errors.append(f"{scene.name}: chưa render (thiếu {renders_dir})")
            continue
        all_errors += check_scene(scene, renders_dir)

    if all_errors:
        print(f"TÌM THẤY {len(all_errors)} LỖI — KHÔNG đóng gói zip:")
        for e in all_errors:
            print(f"  - {e}")
        raise SystemExit(1)

    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            renders_dir = renders_root / scene.name / "renders"
            for pose in read_test_poses(scene.test_poses_csv):
                stem = Path(pose.image_name).stem
                src = renders_dir / f"{stem}.png"
                arcname = f"{scene.name}/{target_filename(pose.image_name, filename_mode)}"
                zf.write(src, arcname)

    n_total = sum(len(read_test_poses(s.test_poses_csv)) for s in scenes)
    print(f"Đã đóng gói {out_path} — {len(scenes)} scene, {n_total} ảnh, chế độ tên file: {filename_mode}")


def verify_zip(zip_path: Path, scenes: list[Scene]):
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    errors = []
    for scene in scenes:
        for pose in read_test_poses(scene.test_poses_csv):
            candidates = [f"{scene.name}/{pose.image_name}", f"{scene.name}/{Path(pose.image_name).stem}.png"]
            if not any(c in names for c in candidates):
                errors.append(f"{scene.name}/{pose.image_name}: không thấy trong zip (đã thử cả 2 kiểu tên)")

    n_total = sum(len(read_test_poses(s.test_poses_csv)) for s in scenes)
    if errors:
        print(f"TÌM THẤY {len(errors)}/{n_total} LỖI trong {zip_path}:")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... và {len(errors) - 50} lỗi khác")
        raise SystemExit(1)
    print(f"{zip_path}: OK — đủ toàn bộ {n_total} ảnh cho {len(scenes)} scene private_set1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission_round1.zip")
    ap.add_argument("--renders_root", default=None, help="Mặc định pipeline/work")
    ap.add_argument("--filename_mode", choices=["literal", "png_ext"], default="literal")
    ap.add_argument("--check_only", default=None, help="Đường dẫn 1 zip có sẵn, chỉ kiểm tra không đóng gói lại")
    args = ap.parse_args()

    scenes = all_scenes("private")  # submission_round1 chỉ nộp cho 8 scene private_set1

    if args.check_only:
        verify_zip(Path(args.check_only), scenes)
        return

    pipeline_root = Path(__file__).resolve().parents[1]
    renders_root = Path(args.renders_root) if args.renders_root else pipeline_root / "work"
    build_zip(Path(args.out), renders_root, args.filename_mode, scenes)
    verify_zip(Path(args.out), scenes)


if __name__ == "__main__":
    main()
