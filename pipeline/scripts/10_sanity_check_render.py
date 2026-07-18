#!/usr/bin/env python3
"""Sanity-check: đường render dùng để NỘP BÀI (04_render_test_poses.py — tự dựng
MiniCam bằng tay từ CSV) có nhất quán với cấu hình lúc TRAIN hay không.

TẠI SAO CẦN SCRIPT NÀY: `04_render_test_poses.py` không dùng `Scene()` gốc của
repo (cái mà train.py tự dùng để load camera) — nó tự dựng camera từ
test_poses.csv vì round 2 không scene nào có ảnh GT thật để load qua Scene(). 2 đường code
xây camera KHÁC NHAU này là đúng chỗ đã từng xảy ra bug thật (antialiasing train
bật nhưng render quên, tự chấm sai ~10 điểm suốt nhiều lần chạy, xem WORKLOG.md —
nhánh feature/depth-anything-v2 round 1 dính lại đúng lỗi này ở thời điểm viết
script này).

CÁCH KIỂM TRA: ảnh TRAIN có GT thật trên đĩa (khác test/ — không scene nào có GT).
Lấy vài pose train thẳng từ COLMAP (`images.bin`/`cameras.bin`), render bằng
ĐÚNG hàm build_minicam()/pipe-stub/antialiasing-detection giống hệt
04_render_test_poses.py, rồi so PSNR/SSIM/LPIPS với ảnh train gốc. Vì Gaussian
được tối ưu trực tiếp trên các ảnh này, PSNR phải RẤT CAO (thường >>test-set,
điển hình 28-40dB+ với 3DGS hội tụ tốt). Nếu ra thấp bất thường (ngang hoặc dưới
mức PSNR đo được trên test-set), gần như chắc chắn có mismatch cấu hình ẩn
(antialiasing, sh_degree, resolution, background color...) — DỪNG LẠI debug
trước khi tin bất kỳ số liệu 05_eval_metrics.py nào khác.

Cách dùng (chạy sau khi đã train + trước khi tin số liệu 05_eval_metrics.py):
    python 10_sanity_check_render.py --scene HCM0421
    python 10_sanity_check_render.py --scene HCM0421 --n_images 10

Yêu cầu: export GS_REPO=/path/to/gaussian-splatting (như 04_render_test_poses.py),
đã train xong scene đó (có gs_model/point_cloud/...), và ảnh train gốc còn trên
đĩa (colmap/dense/images/ HOẶC Dataset/.../train/images/ — script tự tìm cả 2 nơi
vì 03_train_3dgs.sh có thể đã xoá colmap/dense/images/ để dọn đĩa).
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene
from common.poses import TestPose, pose_to_R_T_fov, assert_centered_principal_point

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

GS_REPO = os.environ.get("GS_REPO")
if not GS_REPO or not (Path(GS_REPO) / "train.py").exists():
    raise SystemExit(
        "Chưa set biến môi trường GS_REPO hoặc đường dẫn sai.\n"
        "  export GS_REPO=/path/to/gaussian-splatting\n"
        "(thư mục clone --recursive https://github.com/graphdeco-inria/gaussian-splatting)"
    )
sys.path.insert(0, GS_REPO)

from scene.cameras import MiniCam                              # noqa: E402
from scene.gaussian_model import GaussianModel                  # noqa: E402
from scene.colmap_loader import (                                # noqa: E402
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
)
from gaussian_renderer import render                             # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


class _PipelineParamsStub:
    """Y hệt _PipelineParamsStub của 04_render_test_poses.py — render() chỉ đọc
    đúng 4 field này."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def read_cfg_args(model_dir: Path) -> dict:
    """Y hệt 04_render_test_poses.py::read_cfg_args() — xem docstring ở đó."""
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        return {}
    try:
        ns = eval(cfg_path.read_text(), {"Namespace": Namespace})
        return vars(ns)
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {cfg_path}: {e} — dùng giá trị mặc định/CLI.")
        return {}


def read_pipeline_train_flags(model_dir: Path) -> dict:
    """Y hệt 04_render_test_poses.py::read_pipeline_train_flags() — nguồn đáng
    tin cậy duy nhất cho `antialiasing` thật đã dùng lúc train."""
    p = model_dir / "pipeline_train_flags.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[CẢNH BÁO] Không đọc/parse được {p}: {e} — bỏ qua.")
        return {}


def build_minicam(pose: TestPose, znear: float = 0.01, zfar: float = 100.0) -> MiniCam:
    """Y hệt 04_render_test_poses.py::build_minicam() — PHẢI giống byte-for-byte,
    vì mục đích của script này là kiểm tra chính đường code đó, không phải viết
    lại 1 bản khác rồi tự so với chính nó."""
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


def load_train_poses(sparse_dir: Path) -> dict[str, TestPose]:
    """Đọc images.bin+cameras.bin (hoặc .txt nếu COLMAP xuất dạng text) của
    CHÍNH scene đó, convert sang TestPose — cùng dataclass/convention mà
    common/poses.py dùng cho test_poses.csv (qvec/tvec world->camera, xem
    docstring common/poses.py). Chỉ hỗ trợ camera model PINHOLE/SIMPLE_PINHOLE
    (đúng loại COLMAP xuất ra sau image_undistorter — dữ liệu dense/ luôn đã
    undistort) — báo lỗi rõ thay vì âm thầm đoán sai nếu gặp model khác."""
    bin_images, bin_cameras = sparse_dir / "images.bin", sparse_dir / "cameras.bin"
    txt_images, txt_cameras = sparse_dir / "images.txt", sparse_dir / "cameras.txt"
    if bin_images.exists() and bin_cameras.exists():
        images = read_extrinsics_binary(str(bin_images))
        cameras = read_intrinsics_binary(str(bin_cameras))
    elif txt_images.exists() and txt_cameras.exists():
        images = read_extrinsics_text(str(txt_images))
        cameras = read_intrinsics_text(str(txt_cameras))
    else:
        raise FileNotFoundError(f"Không tìm thấy images.bin/.txt + cameras.bin/.txt trong {sparse_dir}")

    poses: dict[str, TestPose] = {}
    for img in images.values():
        cam = cameras[img.camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params[:3]
            fx = fy = f
        elif cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params[:4]
        else:
            raise ValueError(
                f"{img.name}: camera model '{cam.model}' không được hỗ trợ ở script này "
                f"(chỉ SIMPLE_PINHOLE/PINHOLE — đúng loại dữ liệu dense/ đã undistort). "
                f"Nếu dataset thật dùng model khác, cần bổ sung công thức fx/fy/cx/cy tương ứng."
            )
        poses[img.name] = TestPose(
            image_name=img.name,
            qvec=np.array(img.qvec, dtype=np.float64),
            tvec=np.array(img.tvec, dtype=np.float64),
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=int(cam.width), height=int(cam.height),
        )
    return poses


def find_train_image(scene, image_name: str) -> tuple[Path, bool] | None:
    """03_train_3dgs.sh có thể đã xoá colmap/dense/images/ để dọn đĩa (xem
    CLEANUP_DENSE_IMAGES trong script đó, mặc định BẬT nên trong thực tế HẦU
    NHƯ LUÔN đã bị xoá tới lúc script này chạy) — thử luôn Dataset/.../train/
    images/ gốc (BTC cung cấp, không bao giờ bị xoá) làm phương án dự phòng.

    Trả về (path, is_undistorted). Ảnh gốc ở phương án dự phòng CHƯA undistort
    nên kích thước lệch vài px so với ảnh model thật sự train (đã qua
    image_undistorter) — is_undistorted=False đánh dấu điều đó để main() biết
    cần resize gần đúng thay vì so pixel-exact."""
    pipeline_root = Path(__file__).resolve().parents[1]
    candidates = [
        (pipeline_root / "work" / scene.name / "colmap" / "dense" / "images" / image_name, True),
        (scene.train_images_dir / image_name, False),
    ]
    for c, is_undistorted in candidates:
        if c.exists():
            return c, is_undistorted
    return None


def load_img01(path: Path, resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """resize_to = (W, H) nếu cần ép về đúng kích thước render (chỉ dùng cho
    ảnh gốc CHƯA undistort ở nhánh dự phòng — xem find_train_image)."""
    img = PILImage.open(path).convert("RGB")
    if resize_to is not None and img.size != resize_to:
        img = img.resize(resize_to, PILImage.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model_dir", default=None, help="Mặc định pipeline/work/<scene>/gs_model")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = iteration lớn nhất có sẵn")
    ap.add_argument("--sparse_dir", default=None,
                     help="Mặc định pipeline/work/<scene>/colmap/dense/sparse/0")
    ap.add_argument("--n_images", type=int, default=8, help="Số ảnh train lấy mẫu để kiểm tra")
    ap.add_argument("--white_background", action="store_true")
    args = ap.parse_args()

    scene = get_scene(args.scene)
    pipeline_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir) if args.model_dir else pipeline_root / "work" / scene.name / "gs_model"
    sparse_dir = Path(args.sparse_dir) if args.sparse_dir else (
        pipeline_root / "work" / scene.name / "colmap" / "dense" / "sparse" / "0"
    )

    iteration = args.iteration if args.iteration > 0 else find_latest_iteration(model_dir)
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"

    cfg = read_cfg_args(model_dir)
    train_flags = read_pipeline_train_flags(model_dir)
    sh_degree = cfg.get("sh_degree", 3)
    if "antialiasing" in train_flags:
        antialiasing = bool(train_flags["antialiasing"])
    else:
        antialiasing = False
        print(
            "  [CẢNH BÁO NGHIÊM TRỌNG] Không có pipeline_train_flags.json — không thể biết chắc "
            "antialiasing thật lúc train là gì, đang giả định False. Kết quả sanity-check dưới đây "
            "CHỈ đáng tin nếu giả định này đúng."
        )
    print(f"  Dùng sh_degree={sh_degree}, antialiasing={antialiasing} (giống hệt 04_render_test_poses.py --antialiasing auto)")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(ply_path))
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = _PipelineParamsStub()
    pipe.antialiasing = antialiasing

    all_poses = load_train_poses(sparse_dir)
    names_sorted = sorted(all_poses.keys())
    if not names_sorted:
        raise SystemExit(f"Không đọc được pose train nào từ {sparse_dir}")
    n = min(args.n_images, len(names_sorted))
    step = max(1, len(names_sorted) // n)
    sample_names = names_sorted[::step][:n]

    lpips_fn = None
    if _HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net="alex")
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()
    else:
        print("[CẢNH BÁO] Chưa cài package `lpips` — bỏ qua LPIPS, chỉ tính PSNR/SSIM.")

    rows = []
    n_approx = 0
    print(f"===== {scene.name}: sanity-check {len(sample_names)} ảnh TRAIN qua đúng đường render submission =====")
    for name in sample_names:
        found = find_train_image(scene, name)
        if found is None:
            print(f"  [BỎ QUA] {name}: không tìm thấy ảnh train gốc trên đĩa (đã bị dọn và không có ở Dataset/ gốc?).")
            continue
        gt_path, is_undistorted = found
        pose = all_poses[name]
        assert_centered_principal_point(pose)
        cam = build_minicam(pose)
        with torch.no_grad():
            out = render(cam, gaussians, pipe, background)
        pred = out["render"].clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,3) float01
        render_wh = (pred.shape[1], pred.shape[0])  # PIL dùng (W,H), numpy shape là (H,W,...)

        if is_undistorted:
            gt = load_img01(gt_path)
            if gt.shape != pred.shape:
                print(f"  [BỎ QUA] {name}: kích thước khác nhau GT={gt.shape[:2]} render={pred.shape[:2]} "
                      f"(cả 2 đáng lẽ đã cùng qua undistort — có thể là bug thật, không phải do dọn đĩa).")
                continue
        else:
            # dense/images/ đã bị dọn (bình thường) — ảnh gốc CHƯA undistort nên
            # resize gần đúng về đúng kích thước render để vẫn so được (không
            # pixel-exact, chỉ đủ để bắt sai lệch CẤU HÌNH lớn — mismatch antialiasing/
            # sh_degree làm PSNR tụt nhiều dB, không thể nhầm với sai số resize ~1%).
            gt = load_img01(gt_path, resize_to=render_wh)
            n_approx += 1

        psnr_v = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_v = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
        lpips_v = float("nan")
        if lpips_fn is not None:
            t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            t_pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            if torch.cuda.is_available():
                t_gt, t_pred = t_gt.cuda(), t_pred.cuda()
            with torch.no_grad():
                lpips_v = float(lpips_fn(t_gt, t_pred).item())
        rows.append((name, psnr_v, ssim_v, lpips_v))
        print(f"  [{name}] PSNR={psnr_v:.2f}  SSIM={ssim_v:.4f}  LPIPS={lpips_v:.4f}")

    if not rows:
        raise SystemExit("Không có ảnh nào kiểm tra được — xem lại đường dẫn ảnh train/sparse_dir.")

    arr = np.array([[r[1], r[2]] for r in rows])
    mean_psnr, mean_ssim = arr[:, 0].mean(), arr[:, 1].mean()
    print(f"\n=== TRUNG BÌNH {len(rows)} ảnh TRAIN (qua đúng đường render submission) ===")
    print(f"  PSNR mean={mean_psnr:.2f}  SSIM mean={mean_ssim:.4f}")
    if n_approx:
        print(
            f"  [LƯU Ý] {n_approx}/{len(rows)} ảnh dùng GT XẤP XỈ (resize từ ảnh gốc chưa undistort vì "
            "colmap/dense/images/ đã bị 03_train_3dgs.sh dọn — bình thường, không phải lỗi). PSNR/SSIM "
            "các ảnh này thấp hơn thực tế khoảng 1 chút do resize, không pixel-exact — vẫn đủ tin cậy để "
            "bắt lỗi mismatch cấu hình LỚN (chênh nhiều dB), không dùng số này làm điểm chính thức."
        )
    print(
        "\nCÁCH ĐỌC KẾT QUẢ: đây là ảnh model đã học TRỰC TIẾP, PSNR phải cao RÕ RỆT so với "
        "PSNR đo trên test-set (05_eval_metrics.py) của CHÍNH scene này — điển hình cao hơn "
        "5-15dB+ nếu 3DGS hội tụ bình thường. Nếu PSNR ở đây <= hoặc chỉ nhỉnh hơn chút so với "
        "PSNR test-set, gần như chắc chắn có mismatch cấu hình ẩn giữa train và render (antialiasing, "
        "sh_degree, resolution, background màu nền...) — DỪNG LẠI, kiểm tra lại trước khi tin bất kỳ "
        "số liệu 05_eval_metrics.py nào."
    )


if __name__ == "__main__":
    main()
