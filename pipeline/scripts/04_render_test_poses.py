#!/usr/bin/env python3
"""Render ảnh RGB tại từng pose trong test_poses.csv, dùng model 3DGS đã train
bằng repo GỐC graphdeco-inria/gaussian-splatting.

KHÔNG dùng render.py có sẵn của repo đó — nó chỉ render lại đúng train/test split
COLMAP nội bộ (cần ảnh GT có sẵn trên đĩa để load Scene). Ở đây ta cần render tại
pose TÙY Ý lấy từ test_poses.csv (private_set1 không có ảnh GT), nên tự dựng
camera (MiniCam) trực tiếp, dùng đúng công thức toán mà Camera/MiniCam của repo
dùng (đã đối chiếu source thật: scene/cameras.py, utils/graphics_utils.py,
gaussian_renderer/__init__.py — không suy đoán).

Yêu cầu trước khi chạy:
  export GS_REPO=/path/to/gaussian-splatting   # đã clone --recursive + cài xong
                                                # diff_gaussian_rasterization (cần CUDA)
  python 01_run_colmap.py --scene <scene> ...   # đã có dense/sparse
  bash 03_train_3dgs.sh <scene>                 # đã có gs_model/point_cloud/...

Output: pipeline/work/<scene>/renders/<stem>.png (LUÔN là PNG thật, kể cả nếu
image_name gốc trong CSV có đuôi .JPG). Việc đặt tên file CUỐI CÙNG khi đóng gói
zip nộp bài (giữ đuôi .JPG hay đổi .png) do 06_package_submission.py quyết định
(xem KE_HOACH_VONG1.md mục 4, câu hỏi #3 — vẫn đang chờ xác nhận từ BTC).

Hướng đi Mip-Splatting (Kết quả/Hướng đi.md mục 2, #2): script tự đọc file
`cfg_args` mà train.py ghi lại trong model_dir để biết chính xác `antialiasing`/
`sh_degree` đã dùng lúc train (xem read_cfg_args() bên dưới) — tránh trường hợp
train bật --antialiasing nhưng render quên bật lại (rasterizer sẽ chạy nhưng
kết quả không nhất quán, không hề báo lỗi). Chỉ dùng --antialiasing on/off để ép
thủ công khi thật sự cần so sánh A/B.
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import read_test_poses, pose_to_R_T_fov, assert_centered_principal_point
from common.logging_utils import FileLog

GS_REPO = os.environ.get("GS_REPO")
if not GS_REPO or not (Path(GS_REPO) / "train.py").exists():
    raise SystemExit(
        "Chưa set biến môi trường GS_REPO hoặc đường dẫn sai.\n"
        "  export GS_REPO=/path/to/gaussian-splatting\n"
        "(thư mục clone --recursive https://github.com/graphdeco-inria/gaussian-splatting)"
    )
sys.path.insert(0, GS_REPO)

from scene.cameras import MiniCam                # noqa: E402
from scene.gaussian_model import GaussianModel    # noqa: E402
from gaussian_renderer import render              # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


class _PipelineParamsStub:
    """Thay cho arguments.PipelineParams — render() chỉ đọc đúng 4 field này."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def read_cfg_args(model_dir: Path) -> dict:
    """Đọc file cfg_args mà train.py tự ghi (Namespace(...) dạng str, xem
    train.py::prepare_output_and_logger) để tự phát hiện sh_degree/antialiasing
    ĐÚNG như lúc train — tránh lỗi âm thầm khi train dùng --antialiasing nhưng
    render quên bật lại (hoặc ngược lại), 2 lệnh sẽ ra kết quả không nhất quán
    mà không hề báo lỗi gì. Cùng cách parse mà chính get_combined_args() của repo
    Inria dùng (arguments/__init__.py)."""
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        return {}
    try:
        ns = eval(cfg_path.read_text(), {"Namespace": Namespace})
        return vars(ns)
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {cfg_path}: {e} — dùng giá trị mặc định/CLI.")
        return {}


def build_minicam(pose, znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
    R, T, FovX, FovY = pose_to_R_T_fov(pose)
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=FovX, fovY=FovY
    ).transpose(0, 1).float().cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(pose.width, pose.height, FovY, FovX, znear, zfar,
                    world_view_transform, full_proj_transform)


def find_latest_iteration(model_dir: Path) -> int:
    pc_dir = model_dir / "point_cloud"
    iters = [int(p.name.split("_")[-1]) for p in pc_dir.glob("iteration_*") if p.is_dir()]
    if not iters:
        raise FileNotFoundError(f"Không tìm thấy checkpoint nào trong {pc_dir}")
    return max(iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gs_model")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = iteration lớn nhất có sẵn")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work/<scene>/renders")
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--sh_degree", type=int, default=None,
                     help="Mặc định: tự đọc từ cfg_args (đúng giá trị lúc train). "
                          "Chỉ tự set nếu model không có cfg_args (checkpoint cũ) — khi đó mặc định 3.")
    ap.add_argument("--antialiasing", choices=["auto", "on", "off"], default="auto",
                     help="Mặc định 'auto': tự đọc từ cfg_args — PHẢI khớp giá trị lúc train "
                          "(Hướng đi.md mục 2 #2, xem 03_train_3dgs.sh biến ANTIALIASING). "
                          "Chỉ ép 'on'/'off' thủ công nếu chắc chắn biết mình đang làm gì.")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gs_model"
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work" / scene.name / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration = args.iteration if args.iteration > 0 else find_latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    cfg = read_cfg_args(model_dir)
    if cfg:
        print(f"  cfg_args đọc được: sh_degree={cfg.get('sh_degree')}, antialiasing={cfg.get('antialiasing')}")
    else:
        print("  [CẢNH BÁO] Không có cfg_args trong model_dir (checkpoint train trước khi pipeline hỗ trợ "
              "tự phát hiện) — dùng mặc định sh_degree=3, antialiasing=off trừ khi chỉ định --sh_degree/--antialiasing.")

    sh_degree = args.sh_degree if args.sh_degree is not None else cfg.get("sh_degree", 3)
    if args.antialiasing == "auto":
        antialiasing = bool(cfg.get("antialiasing", False))
    else:
        antialiasing = args.antialiasing == "on"

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = _PipelineParamsStub()
    pipe.antialiasing = antialiasing
    print(f"  Render với sh_degree={sh_degree}, antialiasing={antialiasing}")

    poses = read_test_poses(scene.test_poses_csv)
    log_path = out_dir.parent / "04_render_test_poses.log"
    log = FileLog(log_path)
    print(f"===== {scene.name}: render {len(poses)} pose (iteration {iteration}) -> {out_dir} =====")
    log.write(f"Model: {ply_path}")

    for i, pose in enumerate(poses):
        assert_centered_principal_point(pose)
        cam = build_minicam(pose)
        with torch.no_grad():
            out = render(cam, gaussians, pipe, background)
        img = out["render"].clamp(0, 1).detach().cpu().numpy()  # (3,H,W)
        img = (np.transpose(img, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        if pil_img.size != (pose.width, pose.height):
            raise RuntimeError(
                f"{pose.image_name}: render ra {pil_img.size}, khác yêu cầu "
                f"({pose.width},{pose.height}) — kiểm tra lại MiniCam/getProjectionMatrix."
            )
        stem = Path(pose.image_name).stem
        out_path = out_dir / f"{stem}.png"
        pil_img.save(out_path, format="PNG")
        log.write(f"[{i + 1}/{len(poses)}] {pose.image_name} -> {out_path.name}")

    log.write(f"Xong. {len(poses)} ảnh PNG tại {out_dir}")
    log.close()
    print(f"-> Xong {len(poses)} ảnh. Log chi tiết từng ảnh: {log_path}")


if __name__ == "__main__":
    main()
