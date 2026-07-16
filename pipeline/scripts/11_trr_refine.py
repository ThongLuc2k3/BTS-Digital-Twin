#!/usr/bin/env python3
"""TRR Tier-1 — Tiered Reference-guided Refinement, tầng 1 (plan.md mục 5,
Y_TUONG_TRR_HAU_XU_LY_THAM_CHIEU.md). Hậu xử lý SAU khi 3DGS đã render xong ảnh ở 1
pose bất kỳ (test hoặc holdout) — KHÔNG train model mới, thuần hình học: dùng chính
ảnh train thật + sparse COLMAP đã có pose để "vá" lại vùng render bị mờ/thiếu chi
tiết bằng texture thật đã biết, ở đúng vị trí hình học (không hallucinate).

Thuật toán (đúng 4 bước ở plan.md mục 5):
  1. Với pose cần refine, tìm k ảnh train gần nhất theo vị trí camera + góc nhìn.
  2. Chiếu toàn bộ điểm 3D của sparse vào pose cần refine (pinhole, dùng
     common.poses.qvec2rotmat cho ĐÚNG convention world->camera toàn repo) để biết
     điểm nào rơi vào khung hình + depth — occlusion xử lý bằng cách gộp theo lưới ô
     kích thước patch_size, mỗi ô chỉ giữ điểm gần camera nhất (xấp xỉ z-buffer ở độ
     phân giải patch, đủ dùng cho patch-copy thô, không cần z-buffer đầy đủ).
  3. Với mỗi điểm đại diện còn lại, tra `track` của điểm đó (COLMAP đã biết chính xác
     ảnh train nào quan sát được điểm này, ở toạ độ 2D nào — KHÔNG suy đoán) để lấy
     patch pixel thật quanh toạ độ đó trong ảnh train, chỉ giữ candidate nằm trong k
     ảnh gần nhất đã chọn ở bước 1.
  4. Blend nhiều candidate (nếu có) theo trọng số góc nhìn (cos similarity hướng
     camera train vs hướng camera pose cần refine) × nghịch đảo khoảng cách, rồi
     trộn với ảnh render gốc theo `--blend_alpha` (không thay thế 100% — patch-copy
     bằng translate thô có thể lệch màu/seam, xem giới hạn bên dưới).

GIỚI HẠN CỐ Ý của bản Tier-1 này (ghi rõ để không nhầm là bug):
  - Warp = translate patch thô (KHÔNG affine/homography) — đủ tốt cho pose train/
    render gần nhau (baseline nhỏ), méo dần nếu k-nearest ở xa/góc lệch lớn.
  - Vùng KHÔNG có correspondence nào (floaters/vùng khuất hoàn toàn khỏi mọi ảnh
    train) → GIỮ NGUYÊN pixel render gốc, không sửa — đúng tinh thần "an toàn, không
    hallucinate". Đây là việc của Tier-2 (3DGS-Enhancer, CHƯA làm), không phải của
    script này.
  - Occlusion xấp xỉ theo lưới patch_size, không phải z-buffer per-pixel thật.

Cách dùng:
    python 11_trr_refine.py --scene HCM0421 \\
        --render_dir pipeline/work/HCM0421/holdout_renders \\
        --out_dir pipeline/work/HCM0421/holdout_renders_trr \\
        --poses_csv pipeline/work/HCM0421/holdout/holdout_poses.csv

    python 11_trr_refine.py --scene chair \\
        --render_dir pipeline/work/chair/renders --out_dir pipeline/work/chair/renders_trr
        # (mặc định --poses_csv là test_poses.csv thật của scene)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scenes import get_scene, Scene
from common.poses import read_test_poses, qvec2rotmat, TestPose


def _detect_points2d_scale(rec: "pycolmap.Reconstruction") -> float:
    """QUAN TRỌNG — phát hiện thực nghiệm: `points2D[i].xy` trong sparse round-2 được
    lưu ở ĐỘ PHÂN GIẢI GỐC (trước khi BTC hạ xuống ảnh `train/images/` thật cấp cho
    thí sinh), trong khi `camera.width/height/fx/fy/cx/cy` đã đúng theo ảnh thật đã hạ
    độ phân giải — 2 nguồn LỆCH NHAU đúng hệ số "Scale" ghi trong plan.md mục 2 (chair
    1/1.5, scene HCM* 1/4, bonsai 1/1). Verify: `image.project_point(point3D.xyz)`
    (dùng pose+intrinsics, tự nhất quán với kích thước ảnh thật) lệch `points2D.xy`
    đúng theo tỉ lệ không đổi (std ~0.01px trên hàng trăm điểm) — KHÔNG phải nhiễu/
    outlier, mà là quy ước lưu trữ khác nhau giữa 2 trường dữ liệu trong CÙNG 1 file
    sparse. `point.error` (residual bundle-adjustment nội bộ COLMAP) vẫn nhỏ (~1px) vì
    COLMAP tự nhất quán nội bộ — chỉ lộ ra khi so với intrinsics đã bị BTC rescale.
    KHÔNG ảnh hưởng train/render 3DGS (train.py chỉ dùng pose + points3D.xyz không gian
    3D, không đọc points2D.xy) — chỉ ảnh hưởng chỗ nào tự đọc points2D.xy trực tiếp như
    script này. Đo bằng 1 ảnh có nhiều điểm quan sát nhất để robust, trả về hệ số
    SCALE sao cho `xy_thật = points2D.xy / SCALE`."""
    best_img = max(rec.images.values(), key=lambda im: im.num_points3D)
    obs = best_img.get_observation_points2D()
    ratios = []
    for o in obs:
        proj = best_img.project_point(rec.points3D[o.point3D_id].xyz)
        if proj is None or proj[0] <= 1 or proj[1] <= 1:
            continue
        ratios.append(o.xy[0] / proj[0])
        ratios.append(o.xy[1] / proj[1])
    if not ratios:
        return 1.0
    scale = float(np.median(ratios))
    if abs(scale - 1.0) > 0.02:
        print(f"  [LƯU Ý] points2D.xy lệch camera intrinsics theo hệ số {scale:.4f}x "
              f"(khớp cột 'Scale' của plan.md) — tự chia lại toạ độ patch source.")
    return scale


class SceneRefData:
    """Toàn bộ dữ liệu đọc 1 lần/scene: sparse + camera train + track điểm 3D — tránh
    load lại pycolmap.Reconstruction cho mỗi pose (chậm nếu poses_csv có hàng chục dòng)."""

    def __init__(self, scene: Scene):
        self.rec = pycolmap.Reconstruction(str(scene.provided_sparse_dir))
        self.train_images_dir = scene.train_images_dir
        self.points2d_scale = _detect_points2d_scale(self.rec)

        # Camera train: id -> (tên file, center thế giới, hướng nhìn thế giới) — dùng
        # thẳng projection_center()/viewing_direction() có sẵn của pycolmap (đã verify
        # khớp byte-exact với công thức tự tính qua qvec2rotmat, không suy đoán).
        self.image_id_to_name = {}
        self.train_centers = {}
        self.train_viewdirs = {}
        disk_names = {p.name for p in self.train_images_dir.iterdir()}
        for img_id, image in self.rec.images.items():
            if image.name not in disk_names:
                continue  # ảnh có pose trong sparse nhưng thiếu file thật trên đĩa
            self.image_id_to_name[img_id] = image.name
            self.train_centers[img_id] = image.projection_center()
            self.train_viewdirs[img_id] = image.viewing_direction()

        centers_arr = np.array(list(self.train_centers.values()))
        self.scene_scale = float(np.linalg.norm(centers_arr.max(axis=0) - centers_arr.min(axis=0)))
        if self.scene_scale < 1e-6:
            self.scene_scale = 1.0

        # points3D: mảng xyz song song với id để chiếu hàng loạt bằng numpy, cộng
        # track (ảnh nào thấy điểm này, ở toạ độ 2D nào) để tra cứu sau khi biết điểm
        # nào là đại diện (đã qua occlusion-binning).
        self.point_ids = np.array(list(self.rec.points3D.keys()))
        self.point_xyz = np.array([self.rec.points3D[pid].xyz for pid in self.point_ids])
        self.point_track = {}
        for pid in self.point_ids:
            elements = self.rec.points3D[int(pid)].track.elements
            cand = [(el.image_id, el.point2D_idx) for el in elements if el.image_id in self.image_id_to_name]
            if cand:
                self.point_track[int(pid)] = cand


def pose_center_direction(pose: TestPose) -> tuple[np.ndarray, np.ndarray]:
    """center/hướng nhìn thế giới của pose cần refine — dùng ĐÚNG qvec2rotmat của
    common/poses.py (world->camera) để nhất quán convention với toàn repo, KHÔNG
    dùng lại pose_to_R_T_fov (hàm đó chuyên cho MiniCam gaussian-splatting, có
    transpose khác mục đích ở đây)."""
    R = qvec2rotmat(pose.qvec)  # world -> camera
    t = pose.tvec
    center = -R.T @ t
    viewdir = R.T @ np.array([0.0, 0.0, 1.0])
    return center, viewdir, R, t


def select_k_nearest(query_center, query_viewdir, data: SceneRefData, k: int,
                      angle_weight: float = 1.0) -> set:
    """Xếp hạng ảnh train theo (khoảng cách chuẩn hoá theo scene_scale) + (1-cos góc
    nhìn) — càng nhỏ càng gần/càng cùng hướng. Trả về SET id ảnh train được chọn."""
    scores = []
    for img_id in data.image_id_to_name:
        c = data.train_centers[img_id]
        v = data.train_viewdirs[img_id]
        dist = np.linalg.norm(query_center - c) / data.scene_scale
        cos_sim = float(np.dot(query_viewdir, v))  # cả 2 đã là unit vector
        score = dist + angle_weight * (1.0 - cos_sim)
        scores.append((score, img_id))
    scores.sort(key=lambda x: x[0])
    return {img_id for _, img_id in scores[:k]}


def project_points(data: SceneRefData, R, t, pose: TestPose):
    """Chiếu toàn bộ points3D vào pose cần refine (pinhole thuần). Trả về mảng
    u,v,z,valid_mask (valid = trước camera + rơi trong khung hình)."""
    X_cam = (R @ data.point_xyz.T).T + t  # (N,3)
    z = X_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pose.fx * X_cam[:, 0] / z + pose.cx
        v = pose.fy * X_cam[:, 1] / z + pose.cy
    valid = (z > 1e-3) & (u >= 0) & (u < pose.width) & (v >= 0) & (v < pose.height)
    return u, v, z, valid


def occlusion_binning(u, v, z, valid, patch_size: int, width: int, height: int) -> np.ndarray:
    """Gộp điểm theo lưới ô kích thước patch_size, mỗi ô chỉ giữ điểm gần camera
    nhất (z nhỏ nhất) — xấp xỉ z-buffer ở độ phân giải patch, đủ cho patch-copy thô
    và tự nhiên khớp mật độ 1 patch/ô (không patch nào đè patch khác trong cùng ô).
    Trả về mảng INDEX (vào mảng gốc u/v/z) của các điểm đại diện được chọn."""
    idx_valid = np.nonzero(valid)[0]
    if idx_valid.size == 0:
        return idx_valid
    n_bins_u = width // patch_size + 1
    bin_u = (u[idx_valid] // patch_size).astype(np.int64)
    bin_v = (v[idx_valid] // patch_size).astype(np.int64)
    bin_key = bin_v * n_bins_u + bin_u

    order = np.argsort(z[idx_valid])  # gần camera nhất trước
    bin_key_sorted = bin_key[order]
    _, first_pos = np.unique(bin_key_sorted, return_index=True)
    selected_local = order[first_pos]  # index trong idx_valid, đã là điểm z nhỏ nhất/ô
    return idx_valid[selected_local]


def paste_weighted_patch(accum, weight_buf, ty, tx, source_img, sy, sx, patch_r, weight):
    """Cộng dồn 1 patch (đã nhân weight) từ source_img (quanh sy,sx) vào accum (quanh
    ty,tx) — tự cắt biên sao cho 2 vùng LUÔN cùng shape (lấy margin nhỏ nhất giữa 2
    ảnh + patch_r ở mỗi phía), tránh lỗi shape mismatch ở rìa ảnh."""
    H, W = accum.shape[:2]
    Hs, Ws = source_img.shape[:2]
    top = min(ty, sy, patch_r)
    bottom = min(H - 1 - ty, Hs - 1 - sy, patch_r)
    left = min(tx, sx, patch_r)
    right = min(W - 1 - tx, Ws - 1 - sx, patch_r)
    if top + bottom < 0 or left + right < 0:
        return
    t_patch = accum[ty - top:ty + bottom + 1, tx - left:tx + right + 1]
    w_patch = weight_buf[ty - top:ty + bottom + 1, tx - left:tx + right + 1]
    s_patch = source_img[sy - top:sy + bottom + 1, sx - left:sx + right + 1]
    t_patch += s_patch.astype(np.float64) * weight
    w_patch += weight


def refine_image(render_path: Path, data: SceneRefData, pose: TestPose,
                  k_neighbors: int, patch_size: int, blend_alpha: float,
                  angle_weight: float) -> np.ndarray:
    render = np.asarray(Image.open(render_path).convert("RGB"))
    query_center, query_viewdir, R, t = pose_center_direction(pose)
    neighbor_ids = select_k_nearest(query_center, query_viewdir, data, k_neighbors, angle_weight)

    u, v, z, valid = project_points(data, R, t, pose)
    rep_idx = occlusion_binning(u, v, z, valid, patch_size, pose.width, pose.height)

    accum = np.zeros((pose.height, pose.width, 3), dtype=np.float64)
    weight_buf = np.zeros((pose.height, pose.width), dtype=np.float64)
    patch_r = patch_size // 2

    # Cache ảnh train đã mở (1 pose có thể dùng lại cùng vài ảnh train cho nhiều điểm).
    train_img_cache: dict[str, np.ndarray] = {}

    n_patches_drawn = 0
    for i in rep_idx:
        pid = int(data.point_ids[i])
        cand = data.point_track.get(pid)
        if not cand:
            continue
        cand = [(img_id, p2idx) for img_id, p2idx in cand if img_id in neighbor_ids]
        if not cand:
            continue

        weights = []
        for img_id, _ in cand:
            c = data.train_centers[img_id]
            vd = data.train_viewdirs[img_id]
            dist = np.linalg.norm(query_center - c) / data.scene_scale
            cos_sim = max(0.0, float(np.dot(query_viewdir, vd)))
            weights.append(cos_sim / (1.0 + dist))
        total_w = sum(weights)
        if total_w <= 1e-9:
            continue

        ty, tx = int(round(v[i])), int(round(u[i]))
        for (img_id, p2idx), w in zip(cand, weights):
            if w <= 0:
                continue
            name = data.image_id_to_name[img_id]
            if name not in train_img_cache:
                train_img_cache[name] = np.asarray(
                    Image.open(data.train_images_dir / name).convert("RGB"))
            src = train_img_cache[name]
            xy = data.rec.images[img_id].points2D[p2idx].xy
            # xy lưu ở độ phân giải GỐC (xem _detect_points2d_scale) — chia lại về
            # đúng độ phân giải ảnh train/images/ thật trước khi dùng làm toạ độ patch.
            sx = int(round(xy[0] / data.points2d_scale))
            sy = int(round(xy[1] / data.points2d_scale))
            if not (0 <= sx < src.shape[1] and 0 <= sy < src.shape[0]):
                continue
            paste_weighted_patch(accum, weight_buf, ty, tx, src, sy, sx, patch_r, w / total_w)
        n_patches_drawn += 1

    refined = render.astype(np.float64).copy()
    has_data = weight_buf > 1e-9
    avg_patch = np.zeros_like(refined)
    avg_patch[has_data] = accum[has_data] / weight_buf[has_data, None]
    refined[has_data] = (blend_alpha * avg_patch[has_data] +
                          (1 - blend_alpha) * refined[has_data])
    coverage = float(has_data.mean())
    return np.clip(refined, 0, 255).astype(np.uint8), coverage, n_patches_drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--render_dir", required=True, help="Thư mục ảnh render cần refine (.png)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--poses_csv", default=None, help="Mặc định test_poses.csv thật của scene")
    ap.add_argument("--k_neighbors", type=int, default=5)
    ap.add_argument("--patch_size", type=int, default=9, help="Phải là số lẻ")
    ap.add_argument("--blend_alpha", type=float, default=0.6,
                     help="Trọng số patch tham chiếu khi blend với render gốc (0=giữ nguyên render, 1=thay hẳn)")
    ap.add_argument("--angle_weight", type=float, default=1.0,
                     help="Trọng số phạt lệch góc nhìn khi chọn k ảnh train gần nhất")
    args = ap.parse_args()
    if args.patch_size % 2 == 0:
        raise SystemExit("--patch_size phải là số lẻ (để có tâm patch rõ ràng).")

    scene = get_scene(args.scene)
    poses_csv = Path(args.poses_csv) if args.poses_csv else scene.test_poses_csv
    render_dir = Path(args.render_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang load sparse + track cho {scene.name} ...")
    data = SceneRefData(scene)
    print(f"  -> {len(data.image_id_to_name)} ảnh train, {len(data.point_ids)} điểm 3D "
          f"({len(data.point_track)} điểm có track hợp lệ).")

    poses = read_test_poses(poses_csv)
    for i, pose in enumerate(poses):
        render_path = render_dir / f"{Path(pose.image_name).stem}.png"
        if not render_path.exists():
            print(f"  [BỎ QUA] {pose.image_name}: không có render tại {render_path}")
            continue
        refined, coverage, n_patches = refine_image(
            render_path, data, pose, args.k_neighbors, args.patch_size,
            args.blend_alpha, args.angle_weight)
        out_path = out_dir / render_path.name
        Image.fromarray(refined).save(out_path)
        print(f"  [{i + 1}/{len(poses)}] {pose.image_name}: coverage={coverage:.1%} "
              f"({n_patches} patch) -> {out_path.name}")

    print(f"Xong. Ảnh đã refine tại {out_dir}")


if __name__ == "__main__":
    main()
