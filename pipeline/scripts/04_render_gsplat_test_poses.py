#!/usr/bin/env python3
"""Render ảnh RGB tại từng pose trong test_poses.csv, dùng checkpoint gsplat+MCMC
đã train bằng 03_train_gsplat_mcmc.sh.

KHÔNG dùng eval()/render_traj() có sẵn trong examples/simple_trainer.py của
gsplat — cũng như lý do 04_render_test_poses.py (nhánh Inria) không dùng
render.py gốc: các hàm đó chỉ render lại đúng camera có trong dataset COLMAP đã
load qua Parser, không nhận pose TÙY Ý từ test_poses.csv (private_set1 không có
ảnh GT nên không thể "giả vờ" là 1 ảnh COLMAP có sẵn). Ở đây gọi thẳng
`gsplat.rasterization()` — API cấp thấp của gsplat, không cần đi qua Parser/Dataset
của examples/, nên KHÔNG cần GSPLAT_REPO, chỉ cần `pip install gsplat`.

QUY ƯỚC POSE (đã đối chiếu trực tiếp examples/datasets/colmap.py của gsplat):
  gsplat dùng world-to-camera viewmat = [R | t; 0 0 0 1] với
  R = qvec2rotmat(qvec) TRỰC TIẾP (KHÔNG transpose như quy ước Camera của repo
  Inria — xem common/poses.py::pose_to_R_T_fov, hàm đó CHỈ dùng cho nhánh Inria,
  KHÔNG dùng ở đây). Dùng thẳng qvec2rotmat() + tvec từ common/poses.py.

QUAN TRỌNG — antialiased: gsplat gọi cờ này là `antialiased`/rasterize_mode
("antialiased" | "classic") — KHÁC tên với `antialiasing` của repo Inria. Script
này đọc giá trị THẬT đã dùng lúc train từ `pipeline_train_flags.json` (do
03_train_gsplat_mcmc.sh tự ghi) — không bao giờ tự đoán, đúng bài học từ bug 10
điểm train/render lệch cấu hình trên nhánh Inria (xem 03_train_3dgs.sh).

Checkpoint format (đã đối chiếu trực tiếp examples/simple_trainer.py của gsplat):
  {"step": int, "splats": {"means":[N,3], "scales":[N,3] (log-space),
  "quats":[N,4] (chưa normalize), "opacities":[N] (logit-space),
  "sh0":[N,1,3], "shN":[N,K-1,3]}} tại <result_dir>/ckpts/ckpt_<step>_rank0.pt

Output: pipeline/work/<scene>/renders/<stem>.png — CÙNG format/vị trí với
04_render_test_poses.py (nhánh Inria), để 05_eval_metrics.py và
06_package_submission.py dùng lại y nguyên, không cần sửa gì.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import read_test_poses, qvec2rotmat
from common.logging_utils import FileLog

try:
    from gsplat.rendering import rasterization
except ImportError as e:
    raise SystemExit(
        "Chưa cài package `gsplat` (pip install gsplat) — cần cho render, "
        f"KHÔNG cần GSPLAT_REPO/examples cho việc render. Lỗi gốc: {e}"
    )


def read_pipeline_train_flags(model_dir: Path) -> dict:
    """Đọc pipeline_train_flags.json do 03_train_gsplat_mcmc.sh tự ghi sau khi
    train xong — nguồn đáng tin cậy duy nhất cho antialiased/sh_degree thật đã
    dùng lúc train (xem docstring đầu file)."""
    p = model_dir / "pipeline_train_flags.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {p}: {e} — bỏ qua.")
        return {}


def find_latest_ckpt(model_dir: Path) -> Path:
    ckpt_dir = model_dir / "ckpts"
    ckpts = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"),
                   key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"Không tìm thấy checkpoint nào trong {ckpt_dir}")
    return ckpts[-1]


def load_splats(ckpt_path: Path, device: str = "cuda"):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    raw = ckpt["splats"]
    means = raw["means"].to(device)
    scales = torch.exp(raw["scales"]).to(device)          # log-space -> thật
    quats = F.normalize(raw["quats"], dim=-1).to(device)   # chưa normalize -> unit quaternion
    opacities = torch.sigmoid(raw["opacities"]).to(device)  # logit-space -> [0,1]
    sh0 = raw["sh0"].to(device)
    shN = raw["shN"].to(device)
    colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]
    k = colors.shape[1]
    sh_degree = int(round(math.sqrt(k) - 1))
    if (sh_degree + 1) ** 2 != k:
        raise ValueError(f"Số band SH ({k}) không khớp (sh_degree+1)^2 với sh_degree nguyên nào — checkpoint hỏng?")
    return means, quats, scales, opacities, colors, sh_degree, int(ckpt.get("step", -1))


def build_viewmat_K(pose) -> tuple[torch.Tensor, torch.Tensor]:
    """world-to-camera 4x4 + intrinsics 3x3, ĐÚNG quy ước gsplat (xem docstring
    đầu file) — R = qvec2rotmat(qvec) trực tiếp, KHÔNG transpose."""
    R = qvec2rotmat(pose.qvec)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = pose.tvec
    K = np.array([
        [pose.fx, 0.0, pose.cx],
        [0.0, pose.fy, pose.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return torch.from_numpy(w2c).cuda(), torch.from_numpy(K).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gsplat_model")
    ap.add_argument("--ckpt", default=None, help="Mặc định: checkpoint mới nhất trong <model_dir>/ckpts/")
    ap.add_argument("--out_dir", default=None, help="Mặc định pipeline/work/<scene>/renders")
    ap.add_argument("--antialiased", choices=["auto", "on", "off"], default="auto",
                     help="Mặc định 'auto': tự đọc từ pipeline_train_flags.json — PHẢI khớp giá trị "
                          "lúc train (xem 03_train_gsplat_mcmc.sh biến ANTIALIASED). Chỉ ép 'on'/'off' "
                          "thủ công nếu chắc chắn biết mình đang làm gì.")
    ap.add_argument("--white_background", action="store_true")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gsplat_model"
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_root / "work" / scene.name / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ckpt(model_dir)
    train_flags = read_pipeline_train_flags(model_dir)

    if args.antialiased == "auto":
        if "antialiased" in train_flags:
            antialiased = bool(train_flags["antialiased"])
            print(f"  antialiased đọc từ pipeline_train_flags.json (giá trị THẬT lúc train): {antialiased}")
        else:
            antialiased = False
            print(
                "  [CẢNH BÁO NGHIÊM TRỌNG] Không có pipeline_train_flags.json trong model_dir — "
                "KHÔNG THỂ biết chắc antialiased thật lúc train -> đang mặc định False, CÓ THỂ SAI. "
                "Chạy lại với --antialiased on/off thủ công nếu biết chắc giá trị lúc train."
            )
    else:
        antialiased = args.antialiased == "on"
    rasterize_mode = "antialiased" if antialiased else "classic"

    means, quats, scales, opacities, colors, sh_degree, step = load_splats(ckpt_path)
    print(f"  Checkpoint: {ckpt_path} (step={step}, N_gaussians={means.shape[0]}, sh_degree={sh_degree})")
    print(f"  Render với rasterize_mode={rasterize_mode}")

    bg = 1.0 if args.white_background else 0.0
    background = torch.full((1, 3), bg, dtype=torch.float32, device="cuda")

    poses = read_test_poses(scene.test_poses_csv)
    log_path = out_dir.parent / "04_render_gsplat_test_poses.log"
    log = FileLog(log_path)
    print(f"===== {scene.name}: render {len(poses)} pose (gsplat step {step}) -> {out_dir} =====")
    log.write(f"Checkpoint: {ckpt_path}")

    for i, pose in enumerate(poses):
        # Không cần assert_centered_principal_point() như 04_render_test_poses.py
        # (Inria) — Ks ở đây dùng thẳng cx,cy thật của pose, không có giả định
        # principal point ở giữa ảnh (đó là giới hạn riêng của getProjectionMatrix
        # trong repo Inria, gsplat không có giới hạn này).
        viewmat, K = build_viewmat_K(pose)
        with torch.no_grad():
            render_colors, render_alphas, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                viewmats=viewmat[None], Ks=K[None], width=pose.width, height=pose.height,
                sh_degree=sh_degree, near_plane=0.01, far_plane=100.0,
                rasterize_mode=rasterize_mode, render_mode="RGB",
                backgrounds=background, packed=False,
            )
        img = render_colors[0].clamp(0, 1).detach().cpu().numpy()  # (H,W,3) float01
        img = (img * 255.0).round().astype(np.uint8)
        pil_img = PILImage.fromarray(img)
        if pil_img.size != (pose.width, pose.height):
            raise RuntimeError(
                f"{pose.image_name}: render ra {pil_img.size}, khác yêu cầu "
                f"({pose.width},{pose.height}) — kiểm tra lại build_viewmat_K."
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
