#!/usr/bin/env python3
"""Sinh depth prior (monocular depth) cho 1/nhiều scene, phục vụ depth
regularization của repo graphdeco-inria/gaussian-splatting ("Depth Anything V2").

QUAN TRỌNG — vì sao KHÔNG dùng thẳng `Depth-Anything-V2/run.py` như README của
gaussian-splatting hướng dẫn: đã đối chiếu trực tiếp source thật của
`Depth-Anything-V2/run.py` — script đó LUÔN lưu depth ở dạng ẢNH 8-BIT
(`depth.astype(np.uint8)`, 0-255), trong khi `<GS_REPO>/utils/make_depth_scale.py`
(bước tiếp theo, tính scale/offset khớp với sparse COLMAP) lại đọc depth map với
giả định 16-BIT (`invmonodepthmap.astype(np.float32) / (2**16)`). Nếu dùng thẳng
`run.py` gốc, depth map sẽ bị nén còn 256 mức xám thay vì 65536 — mất rất nhiều
độ chính xác one cách âm thầm, không báo lỗi gì. Script này gọi thẳng
`DepthAnythingV2.infer_image()` (class gốc của Depth-Anything-V2, KHÔNG qua
run.py) rồi tự chuẩn hoá + lưu đúng 16-bit để make_depth_scale.py đọc đúng.

Cách dùng (cần GPU CUDA, đã tự cài Depth-Anything-V2 — xem hướng dẫn cài đặt bên
dưới):
    export DA_REPO=/path/to/Depth-Anything-V2   # đã tải checkpoint vitl vào
                                                  # $DA_REPO/checkpoints/depth_anything_v2_vitl.pth
    export GS_REPO=/path/to/gaussian-splatting   # cần utils/make_depth_scale.py + utils/read_write_model.py
    python 08_generate_depth_priors.py --scene HCM0421
    python 08_generate_depth_priors.py --all --domain bts
    python 08_generate_depth_priors.py --all --domain generic

Cài đặt Depth-Anything-V2 (1 lần, máy có GPU CUDA):
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    cd Depth-Anything-V2
    git checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf   # pin để tái lập (đề bài mục 10.3)
    pip install -r requirements.txt
    mkdir checkpoints && cd checkpoints
    # tải depth_anything_v2_vitl.pth từ
    # https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

Yêu cầu trước khi chạy: đã có `pipeline/work/<scene>/colmap/dense/{images/,sparse/0/}`
(chạy `01_run_colmap.py --scene <scene>` trước — script này đọc trực tiếp ảnh
undistort trong đó, KHÔNG phải ảnh gốc train/images/, để khớp pixel-for-pixel với
sparse/0 dùng ở bước tính scale).

Output:
    pipeline/work/<scene>/colmap/dense/depths_any/<stem>.png   (16-bit, cùng tên gốc
        bỏ đuôi — đúng quy ước make_depth_scale.py cần)
    pipeline/work/<scene>/colmap/dense/sparse/0/depth_params.json   (do
        make_depth_scale.py sinh ra — scale/offset align với sparse COLMAP)

Sau khi chạy xong cho 1 scene, bật depth regularization lúc train bằng:
    DEPTH_PRIOR=1 bash 03_train_3dgs.sh <scene>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene, all_scenes, Scene
from common.logging_utils import FileLog

DA_REPO = os.environ.get("DA_REPO")
GS_REPO = os.environ.get("GS_REPO")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _require_repos() -> None:
    if not DA_REPO or not (Path(DA_REPO) / "depth_anything_v2" / "dpt.py").exists():
        raise SystemExit(
            "Chưa set biến môi trường DA_REPO hoặc đường dẫn sai.\n"
            "  export DA_REPO=/path/to/Depth-Anything-V2\n"
            "(xem hướng dẫn cài đặt ở docstring đầu file này)"
        )
    if not GS_REPO or not (Path(GS_REPO) / "utils" / "make_depth_scale.py").exists():
        raise SystemExit(
            "Chưa set biến môi trường GS_REPO hoặc bản clone không có utils/make_depth_scale.py "
            "(cần checkout đúng commit đã pin — xem 03_train_3dgs.sh đầu file).\n"
            "  export GS_REPO=/path/to/gaussian-splatting"
        )


def load_model(encoder: str, device: str):
    sys.path.insert(0, DA_REPO)
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

    ckpt = Path(DA_REPO) / "checkpoints" / f"depth_anything_v2_{encoder}.pth"
    if not ckpt.exists():
        raise SystemExit(
            f"Không tìm thấy checkpoint {ckpt} — tải về theo hướng dẫn ở docstring đầu file "
            f"(https://huggingface.co/depth-anything/Depth-Anything-V2-Large cho vitl)."
        )
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    return model.to(device).eval()


def generate_depth_maps(scene: Scene, model, input_size: int, log: FileLog) -> Path:
    images_dir = _dense_images_dir(scene)
    if not images_dir.exists():
        raise SystemExit(
            f"{scene.name}: không thấy {images_dir} — chạy 01_run_colmap.py --scene {scene.name} trước."
        )
    out_dir = images_dir.parent / "depths_any"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        raise SystemExit(f"{scene.name}: {images_dir} rỗng — không có ảnh để sinh depth.")

    for i, img_path in enumerate(image_paths):
        raw = cv2.imread(str(img_path))
        if raw is None:
            log.write(f"[BỎ QUA] không đọc được ảnh: {img_path}")
            continue
        with torch.no_grad():
            depth = model.infer_image(raw, input_size)  # float32, inverse-depth tương đối, chưa chuẩn hoá

        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-6:
            log.write(f"[CẢNH BÁO] {img_path.name}: depth phẳng hoàn toàn (min=max={d_min}) — bỏ qua.")
            continue
        depth_norm = (depth - d_min) / (d_max - d_min)  # về [0,1]
        depth_u16 = (depth_norm * 65535.0).round().astype(np.uint16)  # ĐÚNG 16-bit, khớp make_depth_scale.py

        out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), depth_u16)
        log.write(f"[{i + 1}/{len(image_paths)}] {img_path.name} -> {out_path.name} (16-bit)")

    print(f"  -> Đã sinh {len(image_paths)} depth map (16-bit) tại {out_dir}")
    return out_dir


def _dense_images_dir(scene: Scene) -> Path:
    pipeline_root = Path(__file__).resolve().parents[1]
    return pipeline_root / "work" / scene.name / "colmap" / "dense" / "images"


def run_make_depth_scale(scene: Scene, depths_dir: Path, log: FileLog) -> None:
    dense_dir = depths_dir.parent  # .../colmap/dense
    sparse_dir = dense_dir / "sparse" / "0"
    if not (sparse_dir / "cameras.bin").exists():
        raise SystemExit(f"{scene.name}: không thấy {sparse_dir}/cameras.bin — kiểm tra lại 01_run_colmap.py.")

    cmd = [
        sys.executable, str(Path(GS_REPO) / "utils" / "make_depth_scale.py"),
        "--base_dir", str(dense_dir),
        "--depths_dir", str(depths_dir),
        "--model_type", "bin",
    ]
    log.write(f"Chạy: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log.write(result.stdout)
    if result.returncode != 0:
        log.write(result.stderr)
        raise SystemExit(
            f"{scene.name}: make_depth_scale.py thất bại (exit {result.returncode}) — xem log chi tiết:\n"
            f"{result.stderr[-2000:]}"
        )
    params_path = sparse_dir / "depth_params.json"
    if not params_path.exists():
        raise SystemExit(f"{scene.name}: make_depth_scale.py chạy xong nhưng không thấy {params_path}.")
    print(f"  -> Đã sinh {params_path}")


def process_scene(scene: Scene, model, input_size: int) -> None:
    pipeline_root = Path(__file__).resolve().parents[1]
    log_path = pipeline_root / "work" / scene.name / "08_generate_depth_priors.log"
    log = FileLog(log_path)
    print(f"===== {scene.name}: sinh depth prior =====")
    depths_dir = generate_depth_maps(scene, model, input_size, log)
    run_make_depth_scale(scene, depths_dir, log)
    log.close()
    print(f"-> Xong {scene.name}. Bật depth regularization bằng: DEPTH_PRIOR=1 bash 03_train_3dgs.sh {scene.name}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="1 scene cụ thể, vd HCM0421")
    g.add_argument("--all", action="store_true", help="Toàn bộ scene (dùng cùng --domain để lọc)")
    ap.add_argument("--domain", choices=["bts", "generic"], default=None,
                     help="Chỉ dùng cùng --all: lọc theo scene BTS hoặc scene tổng quát (bonsai/chair)")
    ap.add_argument("--encoder", default="vitl", choices=list(MODEL_CONFIGS.keys()),
                     help="Encoder Depth Anything V2 (mặc định vitl theo khuyến nghị README gaussian-splatting)")
    ap.add_argument("--input_size", type=int, default=518, help="Kích thước resize nội bộ khi infer (mặc định repo)")
    args = ap.parse_args()

    _require_repos()

    scenes = [get_scene(args.scene)] if args.scene else all_scenes(args.domain)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[CẢNH BÁO] Không thấy GPU CUDA — chạy Depth Anything V2 trên CPU sẽ RẤT chậm "
              "(hàng chục giây/ảnh thay vì <1s), không khuyến nghị cho >1 scene.")
    print(f"Nạp model Depth Anything V2 ({args.encoder}) lên {device}...")
    model = load_model(args.encoder, device)

    for scene in scenes:
        process_scene(scene, model, args.input_size)


if __name__ == "__main__":
    main()
