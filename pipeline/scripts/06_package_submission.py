#!/usr/bin/env python3
"""Đóng gói submission_round1.zip từ pipeline/work/<scene>/renders/ cho 8 scene
private_set1, kèm kiểm tra tự động TRƯỚC khi nén (đủ scene/ảnh/đúng kích thước) —
đúng checklist ở Hướng đi.md mục 6.

Cách dùng:
    python 06_package_submission.py --out submission_round1.zip
    python 06_package_submission.py --check_only submission_round1.zip   # chỉ kiểm tra zip có sẵn

Về tên file trong zip — BTC (admin AI RACE) đã xác nhận: PHẢI giữ đúng tên/đuôi
trong cột image_name (vd .JPG), KHÔNG được đổi sang .png, nếu không bài nộp không
hợp lệ. Vì vậy:
    --filename_mode literal   (mặc định, BẮT BUỘC dùng): giữ NGUYÊN chuỗi image_name
                                           trong CSV (kể cả đuôi .JPG gốc) — và nội
                                           dung file được MÃ HOÁ LẠI thành đúng định
                                           dạng theo đuôi đó (đuôi .jpg/.jpeg -> JPEG
                                           thật, đuôi .png -> PNG thật). Trước đây
                                           chỉ đổi TÊN mà giữ nguyên bytes PNG gốc —
                                           khiến file .JPG nhưng nội dung vẫn là PNG
                                           (nặng gấp ~4-8 lần JPEG thật, gây vượt hạn
                                           mức 350MB nộp bài) — nay đã sửa.
    --filename_mode png_ext              : đổi đuôi thành .png (giữ để tham khảo,
                                           KHÔNG dùng trừ khi BTC đổi ý — xem cảnh báo
                                           trên).
    --jpeg_quality (mặc định 95)         : chất lượng nén khi đuôi là .jpg/.jpeg.
"""
import argparse
import io
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


def encode_for_arcname(src_png: Path, arcname: str, jpeg_quality: int) -> bytes:
    """Đọc render .png gốc rồi MÃ HOÁ LẠI đúng định dạng theo đuôi thật của arcname
    (.jpg/.jpeg -> JPEG thật, .png -> PNG thật) — không chỉ đổi tên giữ nguyên bytes,
    vì làm vậy tạo ra file .JPG nhưng nội dung PNG (nặng gấp nhiều lần JPEG thật)."""
    ext = Path(arcname).suffix.lower()
    with Image.open(src_png) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        if ext in (".jpg", ".jpeg"):
            im.save(buf, format="JPEG", quality=jpeg_quality)
        elif ext == ".png":
            im.save(buf, format="PNG")
        else:
            raise ValueError(f"Đuôi file không hỗ trợ: {arcname}")
        return buf.getvalue()


def build_zip(out_path: Path, renders_root: Path, filename_mode: str, scenes: list[Scene],
              jpeg_quality: int):
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
                zf.writestr(arcname, encode_for_arcname(src, arcname, jpeg_quality))

    n_total = sum(len(read_test_poses(s.test_poses_csv)) for s in scenes)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Đã đóng gói {out_path} ({size_mb:.1f} MB) — {len(scenes)} scene, {n_total} ảnh, "
          f"chế độ tên file: {filename_mode}, JPEG quality: {jpeg_quality}")


def verify_zip(zip_path: Path, scenes: list[Scene]):
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        errors = []
        for scene in scenes:
            for pose in read_test_poses(scene.test_poses_csv):
                candidates = [f"{scene.name}/{pose.image_name}", f"{scene.name}/{Path(pose.image_name).stem}.png"]
                match = next((c for c in candidates if c in names), None)
                if match is None:
                    errors.append(f"{scene.name}/{pose.image_name}: không thấy trong zip (đã thử cả 2 kiểu tên)")
                    continue
                ext = Path(match).suffix.lower()
                expected_format = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
                with Image.open(io.BytesIO(zf.read(match))) as im:
                    if im.format != expected_format:
                        errors.append(
                            f"{scene.name}/{pose.image_name}: đuôi {ext} nhưng nội dung thật là "
                            f"{im.format} (cần {expected_format}) — kiểm tra lại bước đóng gói."
                        )
                    if im.size != (pose.width, pose.height):
                        errors.append(
                            f"{scene.name}/{pose.image_name}: kích thước {im.size} != "
                            f"yêu cầu ({pose.width},{pose.height})"
                        )

    n_total = sum(len(read_test_poses(s.test_poses_csv)) for s in scenes)
    size_mb = zip_path.stat().st_size / 1e6
    if errors:
        print(f"TÌM THẤY {len(errors)}/{n_total} LỖI trong {zip_path}:")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... và {len(errors) - 50} lỗi khác")
        raise SystemExit(1)
    print(f"{zip_path} ({size_mb:.1f} MB): OK — đủ toàn bộ {n_total} ảnh cho {len(scenes)} scene "
          f"private_set1, đúng định dạng + kích thước.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission_round1.zip")
    ap.add_argument("--renders_root", default=None, help="Mặc định pipeline/work")
    ap.add_argument("--filename_mode", choices=["literal", "png_ext"], default="literal")
    ap.add_argument("--jpeg_quality", type=int, default=95,
                     help="Chất lượng nén JPEG khi đuôi file là .jpg/.jpeg (mặc định 95)")
    ap.add_argument("--check_only", default=None, help="Đường dẫn 1 zip có sẵn, chỉ kiểm tra không đóng gói lại")
    args = ap.parse_args()

    scenes = all_scenes("private")  # submission_round1 chỉ nộp cho 8 scene private_set1

    if args.check_only:
        verify_zip(Path(args.check_only), scenes)
        return

    pipeline_root = Path(__file__).resolve().parents[1]
    renders_root = Path(args.renders_root) if args.renders_root else pipeline_root / "work"
    build_zip(Path(args.out), renders_root, args.filename_mode, scenes, args.jpeg_quality)
    verify_zip(Path(args.out), scenes)


if __name__ == "__main__":
    main()
