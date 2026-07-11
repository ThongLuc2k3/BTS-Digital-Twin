#!/usr/bin/env python3
"""Fine-tune vài nghìn step cuối bằng loss có LPIPS (VGG) + quét lambda_dssim —
xem Kết quả/prompt_76diem.md mục "Fine-tune bằng LPIPS loss ở 3-5k iterations
cuối": loss train mặc định (L1 + lambda*D-SSIM) không tối ưu trực tiếp cho
LPIPS trong khi công thức chấm BTC cho LPIPS trọng số 0.4 (lớn nhất). Thêm term
LPIPS ở giai đoạn cuối, trên random crop để vừa VRAM T4, là "gần như miễn phí"
về GPU-giờ so với train lại từ đầu.

CHẠY SAU 03_train_gsplat_mcmc.sh (cần checkpoint .pt đã có), TRƯỚC
04_render_gsplat_test_poses.py (render bằng checkpoint MỚI, đã fine-tune).

Dùng ĐÚNG `gsplat.rasterization()` với cùng antialiased/sh_degree đã train (đọc
từ pipeline_train_flags.json, không đoán) — tránh tái diễn bug train/render lệch
cấu hình đã từng xảy ra ở nhánh Inria.

Cách dùng:
    python 03b_finetune_lpips_gsplat.py --scene hcm0031
    python 03b_finetune_lpips_gsplat.py --scene hcm0031 --steps 4000 --lambda_dssim 0.3 --lpips_weight 0.5
    python 03b_finetune_lpips_gsplat.py --scene hcm0031 --lambda_dssim 0.2 --lambda_dssim 0.3 --lambda_dssim 0.4
        # (chạy nhiều lần, mỗi lần 1 giá trị — script chỉ nhận 1 --lambda_dssim/lần,
        #  "quét" nghĩa là tự chạy lại script này với các giá trị khác nhau, xem
        #  giao thức kiểm chứng rẻ trong prompt_76diem.md: so trên 2 scene dev cố định)

Output: checkpoint MỚI tại <model_dir>/ckpts/ckpt_<step_gốc + steps>_rank0.pt —
CÙNG format với checkpoint gsplat gốc, 04_render_gsplat_test_poses.py dùng lại
được ngay không cần sửa gì (tự tìm checkpoint mới nhất theo step).
"""
import argparse
import collections
import json
import math
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import TestPose, qvec2rotmat

try:
    from gsplat.rendering import rasterization
except ImportError as e:
    raise SystemExit(f"Chưa cài package `gsplat` (pip install gsplat). Lỗi gốc: {e}")

try:
    import lpips
except ImportError as e:
    raise SystemExit(f"Chưa cài package `lpips` (pip install lpips) — bắt buộc cho script này. Lỗi gốc: {e}")


# --- đọc images.bin/cameras.bin bằng struct, tự vendor — KHÔNG phụ thuộc repo
# Inria (GS_REPO) hay hàm nội bộ (có thể không public/đổi tên) của gsplat, vì
# nhánh gsplat này không nên cần cài thêm repo khác chỉ để đọc COLMAP. Format
# nhị phân theo đúng chuẩn công khai của COLMAP (scripts/python/read_write_model.py
# chính thức, cùng format mà scene/colmap_loader.py của repo Inria cũng đọc —
# qvec là (qw,qx,qy,qz), cùng quy ước với qvec2rotmat() ở common/poses.py).

_CameraModel = collections.namedtuple("_CameraModel", ["model_id", "model_name", "num_params"])
_CAMERA_MODELS = {
    0: _CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: _CameraModel(1, "PINHOLE", 4),
}
_Image = collections.namedtuple("_Image", ["id", "qvec", "tvec", "camera_id", "name"])
_Camera = collections.namedtuple("_Camera", ["id", "model", "width", "height", "params"])


def _read_bytes(fid, num_bytes, fmt):
    data = fid.read(num_bytes)
    return struct.unpack(fmt, data)


def _read_cameras_bin(path: Path) -> dict[int, _Camera]:
    cameras = {}
    with open(path, "rb") as fid:
        (num_cameras,) = _read_bytes(fid, 8, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_bytes(fid, 24, "<iiQQ")
            model = _CAMERA_MODELS.get(model_id)
            if model is None:
                raise ValueError(
                    f"camera_id={camera_id}: model_id={model_id} không phải SIMPLE_PINHOLE/PINHOLE "
                    f"(2 model duy nhất script này hỗ trợ — đúng loại dữ liệu dense/ đã undistort)."
                )
            params = _read_bytes(fid, 8 * model.num_params, "<" + "d" * model.num_params)
            cameras[camera_id] = _Camera(camera_id, model.model_name, width, height, params)
    return cameras


def _read_images_bin(path: Path) -> dict[int, _Image]:
    images = {}
    with open(path, "rb") as fid:
        (num_reg_images,) = _read_bytes(fid, 8, "<Q")
        for _ in range(num_reg_images):
            props = _read_bytes(fid, 64, "<idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            name = b""
            while True:
                (c,) = _read_bytes(fid, 1, "<c")
                if c == b"\x00":
                    break
                name += c
            (num_points2d,) = _read_bytes(fid, 8, "<Q")
            fid.read(24 * num_points2d)  # (x, y, point3D_id) mỗi điểm — không cần cho fine-tune
            images[image_id] = _Image(image_id, qvec, tvec, camera_id, name.decode("utf-8"))
    return images


def load_train_poses(sparse_dir: Path) -> dict[str, TestPose]:
    images = _read_images_bin(sparse_dir / "images.bin")
    cameras = _read_cameras_bin(sparse_dir / "cameras.bin")
    poses: dict[str, TestPose] = {}
    for img in images.values():
        cam = cameras[img.camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        else:  # PINHOLE
            fx, fy, cx, cy = cam.params[:4]
        poses[img.name] = TestPose(
            image_name=img.name, qvec=img.qvec, tvec=img.tvec,
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=int(cam.width), height=int(cam.height),
        )
    return poses


def find_train_image(scene, image_name: str) -> Path | None:
    pipeline_root = Path(__file__).resolve().parents[1]
    for c in (
        pipeline_root / "work" / scene.name / "colmap" / "dense" / "images" / image_name,
        scene.train_images_dir / image_name,
    ):
        if c.exists():
            return c
    return None


def find_latest_ckpt(model_dir: Path) -> Path:
    ckpts = sorted((model_dir / "ckpts").glob("ckpt_*_rank0.pt"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"Không tìm thấy checkpoint nào trong {model_dir}/ckpts")
    return ckpts[-1]


def build_viewmat_K(pose: TestPose):
    R = qvec2rotmat(pose.qvec)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = pose.tvec
    K = np.array([[pose.fx, 0, pose.cx], [0, pose.fy, pose.cy], [0, 0, 1]], dtype=np.float32)
    return torch.from_numpy(w2c).cuda(), torch.from_numpy(K).cuda()


def gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d.expand(3, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM chuẩn (Gaussian window 11x11, sigma=1.5) trên ảnh (1,3,H,W) trong [0,1] —
    công thức y hệt bản dùng phổ biến trong 3DGS/pytorch-msssim."""
    device = img1.device
    window = gaussian_window(window_size, 1.5, device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=3) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def random_crop(gt: torch.Tensor, pred: torch.Tensor, size: int):
    """gt,pred: (3,H,W) trong [0,1]. Trả về crop vuông cùng vị trí ở cả 2 ảnh."""
    _, h, w = gt.shape
    size = min(size, h, w)
    y0 = random.randint(0, h - size)
    x0 = random.randint(0, w - size)
    return gt[:, y0:y0 + size, x0:x0 + size], pred[:, y0:y0 + size, x0:x0 + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gsplat_model")
    ap.add_argument("--sparse_dir", default=None, help="Mặc định pipeline/work/<scene>/colmap/dense/sparse/0")
    ap.add_argument("--steps", type=int, default=4000, help="Số step fine-tune thêm (khuyến nghị 3000-5000)")
    ap.add_argument("--lambda_dssim", type=float, default=0.2,
                     help="Trọng số D-SSIM trong loss chính (mặc định 0.2 như 3DGS gốc; "
                          "thử 0.3-0.4 vì SSIM chiếm 0.3 điểm số BTC)")
    ap.add_argument("--lpips_weight", type=float, default=0.3,
                     help="Trọng số term LPIPS thêm vào (0 = tắt hẳn, chỉ còn L1+D-SSIM)")
    ap.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex"],
                     help="Mặc định vgg (đúng đề xuất trong prompt_76diem.md — perceptual tốt hơn alex cho fine-tune)")
    ap.add_argument("--crop_size", type=int, default=320, help="Kích thước crop vuông (256-384px để vừa VRAM T4)")
    ap.add_argument("--lr_scale", type=float, default=0.1,
                     help="Nhân LR gốc của gsplat với hệ số này cho giai đoạn fine-tune "
                          "(model đã hội tụ, LR đầy đủ dễ phá vỡ kết quả tốt đã có)")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gsplat_model"
    sparse_dir = Path(args.sparse_dir) if args.sparse_dir else pipeline_root / "work" / scene.name / "colmap" / "dense" / "sparse" / "0"

    ckpt_path = find_latest_ckpt(model_dir)
    ckpt = torch.load(str(ckpt_path), map_location="cuda", weights_only=True)
    raw = ckpt["splats"]
    base_step = int(ckpt.get("step", 0))
    print(f"  Nạp checkpoint: {ckpt_path} (step gốc={base_step})")

    # Tham số RAW (log/logit-space) — fine-tune trực tiếp trên đúng parametrization
    # mà gsplat dùng lúc train, không phải giá trị đã activate.
    means = raw["means"].cuda().clone().requires_grad_(True)
    scales_raw = raw["scales"].cuda().clone().requires_grad_(True)
    quats_raw = raw["quats"].cuda().clone().requires_grad_(True)
    opacities_raw = raw["opacities"].cuda().clone().requires_grad_(True)
    sh0 = raw["sh0"].cuda().clone().requires_grad_(True)
    shN = raw["shN"].cuda().clone().requires_grad_(True)
    k = sh0.shape[1] + shN.shape[1]
    sh_degree = int(round(math.sqrt(k) - 1))

    train_flags_path = model_dir / "pipeline_train_flags.json"
    train_flags = json.loads(train_flags_path.read_text()) if train_flags_path.exists() else {}
    antialiased = bool(train_flags.get("antialiased", False))
    if not train_flags_path.exists():
        print("  [CẢNH BÁO NGHIÊM TRỌNG] Không có pipeline_train_flags.json — không biết chắc "
              "antialiased thật lúc train, giả định False. Có thể làm sai lệch fine-tune.")
    rasterize_mode = "antialiased" if antialiased else "classic"
    print(f"  sh_degree={sh_degree}, rasterize_mode={rasterize_mode} (đọc từ checkpoint/train_flags, không đoán)")

    # LR nhỏ hơn nhiều so với lúc train chính (model đã hội tụ) — theo đúng tỉ lệ
    # tương đối giữa các nhóm tham số mà create_splats_with_optimizers của gsplat dùng
    # (means_lr thấp nhất, sh_lr thấp hơn nữa), chỉ scale đều bằng --lr_scale cho đơn giản.
    lr = args.lr_scale
    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4 * lr},
        {"params": [scales_raw], "lr": 5e-3 * lr},
        {"params": [quats_raw], "lr": 1e-3 * lr},
        {"params": [opacities_raw], "lr": 5e-2 * lr},
        {"params": [sh0], "lr": 2.5e-3 * lr},
        {"params": [shN], "lr": (2.5e-3 / 20) * lr},
    ])

    lpips_fn = lpips.LPIPS(net=args.lpips_net).cuda()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    poses = load_train_poses(sparse_dir)
    names = list(poses.keys())
    gt_cache: dict[str, Path] = {}
    for name in names:
        p = find_train_image(scene, name)
        if p is not None:
            gt_cache[name] = p
    names = [n for n in names if n in gt_cache]
    if not names:
        raise SystemExit("Không tìm thấy ảnh train nào trên đĩa — kiểm tra lại colmap/dense/images hoặc Dataset gốc.")
    print(f"  {len(names)} ảnh train dùng để fine-tune, {args.steps} steps, "
          f"lambda_dssim={args.lambda_dssim}, lpips_weight={args.lpips_weight}, crop={args.crop_size}")

    background = torch.zeros(1, 3, dtype=torch.float32, device="cuda")

    for step in range(1, args.steps + 1):
        name = random.choice(names)
        pose = poses[name]
        gt_full = np.asarray(PILImage.open(gt_cache[name]).convert("RGB")).astype(np.float32) / 255.0
        gt_full_t = torch.from_numpy(gt_full).permute(2, 0, 1).cuda()  # (3,H,W)

        viewmat, K = build_viewmat_K(pose)
        colors = torch.cat([sh0, shN], dim=1)
        scales = torch.exp(scales_raw)
        quats = F.normalize(quats_raw, dim=-1)
        opacities = torch.sigmoid(opacities_raw)

        render_colors, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmat[None], Ks=K[None], width=pose.width, height=pose.height,
            sh_degree=sh_degree, near_plane=0.01, far_plane=100.0,
            rasterize_mode=rasterize_mode, render_mode="RGB",
            backgrounds=background, packed=False,
        )
        pred_full_t = render_colors[0].permute(2, 0, 1).clamp(0, 1)  # (3,H,W)

        gt_crop, pred_crop = random_crop(gt_full_t, pred_full_t, args.crop_size)
        l1 = F.l1_loss(pred_crop, gt_crop)
        ssim_v = ssim(pred_crop.unsqueeze(0), gt_crop.unsqueeze(0))
        loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ssim_v)
        if args.lpips_weight > 0:
            lp = lpips_fn(pred_crop.unsqueeze(0) * 2 - 1, gt_crop.unsqueeze(0) * 2 - 1).mean()
            loss = loss + args.lpips_weight * lp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == args.steps:
            print(f"  [step {step}/{args.steps}] loss={loss.item():.4f} l1={l1.item():.4f} ssim={ssim_v.item():.4f}")

    new_step = base_step + args.steps
    new_splats = {
        "means": means.detach(), "scales": scales_raw.detach(), "quats": quats_raw.detach(),
        "opacities": opacities_raw.detach(), "sh0": sh0.detach(), "shN": shN.detach(),
    }
    out_path = model_dir / "ckpts" / f"ckpt_{new_step}_rank0.pt"
    torch.save({"step": new_step, "splats": new_splats}, out_path)
    print(f"-> Đã lưu checkpoint fine-tune: {out_path}")
    print("   (04_render_gsplat_test_poses.py sẽ tự chọn checkpoint step lớn nhất -> dùng bản này)")


if __name__ == "__main__":
    main()
