#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

GS_REPO = os.environ.get("GS_REPO")
if not GS_REPO or not (Path(GS_REPO) / "train.py").exists():
    raise SystemExit("Chưa set GS_REPO hoặc GS_REPO sai.")
sys.path.insert(0, GS_REPO)

from scene.cameras import MiniCam  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402


@dataclass
class TestPose:
    image_name: str
    qvec: np.ndarray
    tvec: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class PipelineParamsStub:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def read_test_poses(csv_path: Path) -> list[TestPose]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(TestPose(
                image_name=r["image_name"],
                qvec=np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])]),
                tvec=np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])]),
                fx=float(r["fx"]),
                fy=float(r["fy"]),
                cx=float(r["cx"]),
                cy=float(r["cy"]),
                width=int(float(r["width"])),
                height=int(float(r["height"])),
            ))
    return rows


def pose_to_R_T_fov(pose: TestPose):
    Rmat = qvec2rotmat(pose.qvec)
    R = Rmat.transpose()
    T = pose.tvec.copy()
    fov_x = focal2fov(pose.fx, pose.width)
    fov_y = focal2fov(pose.fy, pose.height)
    return R, T, fov_x, fov_y


def build_minicam(pose: TestPose, znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
    R, T, fov_x, fov_y = pose_to_R_T_fov(pose)
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y
    ).transpose(0, 1).float().cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(
        pose.width,
        pose.height,
        fov_y,
        fov_x,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    )


def read_cfg_args(model_dir: Path) -> dict:
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        return {}
    return vars(eval(cfg_path.read_text(), {"Namespace": Namespace}))


def read_pipeline_train_flags(model_dir: Path) -> dict:
    p = model_dir / "pipeline_train_flags.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def find_latest_iteration(model_dir: Path) -> int:
    iters = [int(p.name.split("_")[-1]) for p in (model_dir / "point_cloud").glob("iteration_*") if p.is_dir()]
    if not iters:
        raise FileNotFoundError(f"Không có checkpoint trong {model_dir / 'point_cloud'}")
    return max(iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    scene_root = Path(args.dataset_root) / args.scene
    poses_csv = scene_root / "test" / "test_poses.csv"
    if not poses_csv.exists():
        raise SystemExit(f"Không thấy {poses_csv}")

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration = args.iteration if args.iteration > 0 else find_latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise SystemExit(f"Không thấy {ply_path}")

    cfg = read_cfg_args(model_dir)
    flags = read_pipeline_train_flags(model_dir)
    sh_degree = cfg.get("sh_degree", 3)
    antialiasing = bool(flags.get("antialiasing", False))

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipe = PipelineParamsStub()
    pipe.antialiasing = antialiasing

    poses = read_test_poses(poses_csv)
    print(f"Render {len(poses)} pose | scene={args.scene} | iteration={iteration} | sh_degree={sh_degree} | antialiasing={antialiasing}")
    for i, pose in enumerate(poses, start=1):
        cam = build_minicam(pose)
        with torch.no_grad():
            out = render(cam, gaussians, pipe, bg)
        img = out["render"].clamp(0, 1).detach().cpu().numpy()
        img = (np.transpose(img, (1, 2, 0)) * 255.0).round().astype(np.uint8)
        out_path = out_dir / f"{Path(pose.image_name).stem}.png"
        Image.fromarray(img).save(out_path, format="PNG")
        if i % 10 == 0 or i == len(poses):
            print(f"  {i}/{len(poses)} -> {out_path.name}")


if __name__ == "__main__":
    main()
