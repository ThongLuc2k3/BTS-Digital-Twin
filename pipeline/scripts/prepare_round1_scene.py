#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import pycolmap


def find_missing_images(rec: "pycolmap.Reconstruction", images_dir: Path) -> tuple[list[str], list[str]]:
    all_names = sorted(image.name for image in rec.images.values())
    valid_names = [n for n in all_names if (images_dir / n).exists()]
    missing_names = sorted(set(all_names) - set(valid_names))
    return valid_names, missing_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_root", required=True, help=".../VAI_NVS_DATA/phase1/public_set")
    ap.add_argument("--work_root", required=True, help="pipeline/work")
    args = ap.parse_args()

    scene_root = Path(args.dataset_root) / args.scene
    images_dir = scene_root / "train" / "images"
    sparse_dir = scene_root / "train" / "sparse" / "0"
    if not images_dir.is_dir():
        raise SystemExit(f"Không thấy {images_dir}")
    if not sparse_dir.is_dir():
        raise SystemExit(f"Không thấy {sparse_dir}")

    work_root = Path(args.work_root)
    colmap_root = work_root / args.scene / "colmap"
    dense_dir = colmap_root / "dense"
    filtered_sparse_dir = colmap_root / "_sparse_filtered"

    rec = pycolmap.Reconstruction(sparse_dir)
    valid_names, missing_names = find_missing_images(rec, images_dir)
    if missing_names:
        missing_set = set(missing_names)
        for image in list(rec.images.values()):
            if image.name in missing_set:
                rec.deregister_frame(image.frame_id)
        shutil.rmtree(filtered_sparse_dir, ignore_errors=True)
        filtered_sparse_dir.mkdir(parents=True, exist_ok=True)
        rec.write_binary(filtered_sparse_dir)
        input_sparse_dir = filtered_sparse_dir
    else:
        input_sparse_dir = sparse_dir

    shutil.rmtree(dense_dir, ignore_errors=True)
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=input_sparse_dir,
        image_path=images_dir,
        output_type="COLMAP",
    )

    flat_sparse = dense_dir / "sparse"
    nested_sparse = flat_sparse / "0"
    if flat_sparse.exists() and not nested_sparse.exists():
        tmp = dense_dir / "_sparse_tmp"
        flat_sparse.rename(tmp)
        nested_parent = dense_dir / "sparse"
        nested_parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(nested_parent / "0")

    final_sparse = dense_dir / "sparse" / "0"
    final_images = dense_dir / "images"
    if not final_sparse.is_dir():
        raise SystemExit(f"Không tạo được {final_sparse}")
    if not final_images.is_dir():
        raise SystemExit(f"Không tạo được {final_images}")

    print(f"Prepared scene: {args.scene}")
    print(f"Source sparse : {sparse_dir}")
    print(f"Dense output  : {dense_dir}")
    print(f"Valid images  : {len(valid_names)}")
    print(f"Missing images: {len(missing_names)}")
    if missing_names:
        print("Missing list  :", missing_names)


if __name__ == "__main__":
    main()
