#
# compact_gaussian.py
#
# Bổ sung "Learnable Gaussian Volume Mask" của bài báo:
#   Lee et al., "Compact 3D Gaussian Representation for Radiance Field",
#   arXiv:2311.13681 (CVPR 2024) — mục 3.1 "Gaussian Volume Mask".
#
# Ý tưởng gốc của bài báo: gắn thêm 1 tham số học được m_n cho mỗi Gaussian,
# dùng straight-through estimator để tạo mask nhị phân M_n in {0,1} nhân vào
# scale và opacity. Gaussian có M_n = 0 (không đóng góp) sẽ bị prune ở mỗi
# đợt densification. Loss phụ L_m = mean(sigmoid(m_n)) ép mask co về 0 cho
# các Gaussian dư thừa -> giảm số lượng Gaussian mà không giảm chất lượng.
#
# Bổ sung THÊM theo yêu cầu đề bài (không có trong bài báo gốc):
#   "Giữ Gaussian ở các vùng có độ chi tiết cao (ăng-ten, RRU, cáp)."
#   -> Lớp DetailRegionSet cho phép khai báo các hộp bao (bounding box) 3D
#      quanh những cụm chi tiết mảnh/nhỏ (ăng-ten, RRU, cáp...). Các Gaussian
#      rơi vào trong hộp sẽ:
#        (a) được giảm trọng số của loss mask L_m (ít bị ép về 0 hơn), và
#        (b) (tuỳ chọn) được BẢO VỆ CỨNG — không bao giờ bị prune bởi mask,
#            bất kể giá trị mask học được là bao nhiêu.
#
# File này được thiết kế để đặt cùng cấp với train.py trong repo
# graphdeco-inria/gaussian-splatting (không sửa file gốc), rồi import vào
# 1 bản train.py mới (xem train_compact.py đi kèm).

import json
from typing import List, Optional

import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid


# --------------------------------------------------------------------------
# 1) Vùng chi tiết cao (ăng-ten / RRU / cáp...) khai báo bằng hộp bao 3D
# --------------------------------------------------------------------------
class DetailRegionSet:
    """Tập hợp các hộp bao AABB (axis-aligned bounding box) trong không gian
    COLMAP/3DGS (cùng hệ toạ độ với sparse point cloud và Gaussian). Dùng để
    đánh dấu các cụm chi tiết mảnh (ăng-ten, RRU, cáp, giá đỡ...) trên trụ BTS
    mà bạn muốn hạn chế bị cắt tỉa bởi mask nén.

    Định dạng file JSON, ví dụ xem detail_regions_example.json:
    [
      {"name": "anten_1", "min": [x0, y0, z0], "max": [x1, y1, z1]},
      {"name": "rru_1",   "min": [...],        "max": [...]},
      {"name": "cap_chinh","min": [...],       "max": [...], "protect_weight": 0.02}
    ]

    "protect_weight" (tuỳ chọn, mặc định = protect_weight_default của set)
    càng NHỎ thì Gaussian trong hộp càng ít bị ép mask về 0 (0 = gần như miễn
    nhiễm với loss mask).
    """

    def __init__(self, boxes: List[dict], protect_weight_default: float = 0.05,
                 margin: float = 0.0, device: str = "cuda"):
        assert len(boxes) > 0, "DetailRegionSet rỗng — cần ít nhất 1 hộp bao."
        mins, maxs, weights = [], [], []
        for b in boxes:
            bmin = torch.tensor(b["min"], dtype=torch.float32) - margin
            bmax = torch.tensor(b["max"], dtype=torch.float32) + margin
            mins.append(bmin)
            maxs.append(bmax)
            weights.append(float(b.get("protect_weight", protect_weight_default)))
        self.mins = torch.stack(mins).to(device)          # (K,3)
        self.maxs = torch.stack(maxs).to(device)          # (K,3)
        self.weights = torch.tensor(weights, device=device)  # (K,)
        self.names = [b.get("name", f"box{i}") for i, b in enumerate(boxes)]

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "DetailRegionSet":
        with open(path, "r", encoding="utf-8") as f:
            boxes = json.load(f)
        return cls(boxes, **kwargs)

    def _inside_per_box(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (N,3) -> trả về (N,K) bool: điểm n có nằm trong hộp k không
        xyz = xyz.to(self.mins.device)
        ge_min = (xyz.unsqueeze(1) >= self.mins.unsqueeze(0)).all(dim=-1)  # (N,K)
        le_max = (xyz.unsqueeze(1) <= self.maxs.unsqueeze(0)).all(dim=-1)  # (N,K)
        return ge_min & le_max

    def inside(self, xyz: torch.Tensor) -> torch.Tensor:
        """(N,) bool — True nếu điểm nằm trong BẤT KỲ hộp chi tiết nào."""
        return self._inside_per_box(xyz).any(dim=-1)

    def weight(self, xyz: torch.Tensor) -> torch.Tensor:
        """(N,) float — trọng số nhân vào loss mask L_m cho từng Gaussian.
        = 1.0 nếu ở ngoài mọi hộp (áp dụng nén như bình thường).
        = protect_weight của hộp (nhỏ hơn 1) nếu rơi vào 1 hoặc nhiều hộp
          (lấy giá trị NHỎ NHẤT trong các hộp chứa điểm đó -> bảo vệ mạnh nhất).
        """
        inside = self._inside_per_box(xyz)  # (N,K)
        N = xyz.shape[0]
        w = torch.ones(N, device=self.mins.device)
        if inside.any():
            # với mỗi điểm, lấy min(protect_weight) trong các box chứa nó
            masked_w = torch.where(inside, self.weights.unsqueeze(0).expand(N, -1),
                                    torch.full_like(self.weights.unsqueeze(0).expand(N, -1), float("inf")))
            best = masked_w.min(dim=-1).values
            w = torch.where(torch.isfinite(best), best, w)
        return w


# --------------------------------------------------------------------------
# 2) GaussianModel có thêm mask nén học được (Compact-3DGS, mục 3.1)
# --------------------------------------------------------------------------
class CompactGaussianModel(GaussianModel):
    """Kế thừa GaussianModel gốc, chỉ thêm 1 tham số `_mask` (N,1) học được
    cùng cơ chế mask nhị phân (straight-through estimator) áp lên scale và
    opacity, đúng công thức (1)-(2) trong bài báo Compact-3DGS. Không đụng gì
    tới densification theo gradient hay opacity-reset gốc của 3DGS — 2 cơ chế
    chạy song song, bổ sung cho nhau (đúng như trong Fig.3 của bài báo).
    """

    def __init__(self, sh_degree, optimizer_type="default",
                 mask_threshold: float = 0.01, mask_init: float = 0.9):
        super().__init__(sh_degree, optimizer_type)
        self._mask = torch.empty(0)
        self.mask_threshold = mask_threshold   # epsilon trong công thức (1)
        self.mask_init = mask_init             # sigmoid(m_init) ~ 0.9 lúc khởi tạo

    # ---- khởi tạo mask khi tạo Gaussian từ sparse point cloud ----
    def create_from_pcd(self, pcd, cam_infos, spatial_lr_scale):
        super().create_from_pcd(pcd, cam_infos, spatial_lr_scale)
        n = self.get_xyz.shape[0]
        init_logit = inverse_sigmoid(torch.full((n, 1), self.mask_init, device="cuda"))
        self._mask = nn.Parameter(init_logit.requires_grad_(True))

    # ---- khi load lại 1 file .ply đã train xong (vd để "nén hậu kỳ" 1 model
    #      3DGS gốc đã train sẵn) thì khởi tạo mask mới toanh cho nó ----
    def load_ply(self, path, use_train_test_exp=False):
        super().load_ply(path, use_train_test_exp)
        n = self.get_xyz.shape[0]
        init_logit = inverse_sigmoid(torch.full((n, 1), self.mask_init, device="cuda"))
        self._mask = nn.Parameter(init_logit.requires_grad_(True))

    # ---- capture/restore checkpoint: thêm _mask vào cuối tuple, tương thích
    #      ngược với checkpoint không có mask (len(model_args) == 12) ----
    def capture(self):
        base = super().capture()
        return base + (self._mask,)

    def restore(self, model_args, training_args):
        if len(model_args) == 13:
            *base_args, mask = model_args
            super().restore(tuple(base_args), training_args)
            self._mask = mask.cuda()
            # gán lại param group "mask" bằng đúng tensor đã restore
            for group in self.optimizer.param_groups:
                if group["name"] == "mask":
                    group["params"][0] = nn.Parameter(self._mask.requires_grad_(True))
                    self._mask = group["params"][0]
        else:
            # checkpoint cũ (không có mask) -> restore bình thường rồi init mask mới
            super().restore(model_args, training_args)
            n = self.get_xyz.shape[0]
            init_logit = inverse_sigmoid(torch.full((n, 1), self.mask_init, device="cuda"))
            self._mask = nn.Parameter(init_logit.requires_grad_(True))
            self.optimizer.add_param_group({'params': [self._mask], 'lr': 1e-2, 'name': 'mask'})

    # ---- thêm nhóm tham số "mask" vào optimizer cùng lúc với các nhóm gốc ----
    def training_setup(self, training_args):
        super().training_setup(training_args)
        mask_lr = getattr(training_args, "mask_lr", 1e-2)
        self.optimizer.add_param_group({'params': [self._mask], 'lr': mask_lr, 'name': 'mask'})

    # ---- mask nhị phân straight-through (công thức 1) ----
    @property
    def get_mask_binary(self) -> torch.Tensor:
        m = torch.sigmoid(self._mask)
        hard = (m > self.mask_threshold).float()
        # straight-through estimator: forward = hard, backward = gradient của m
        return hard.detach() - m.detach() + m

    # ---- scale & opacity bị nhân mask trước khi đưa vào rasterizer
    #      (render() trong gaussian_renderer/__init__.py gọi pc.get_scaling
    #      và pc.get_opacity, không cần sửa gì ở renderer) ----
    @property
    def get_scaling(self) -> torch.Tensor:
        return super().get_scaling * self.get_mask_binary

    @property
    def get_opacity(self) -> torch.Tensor:
        return super().get_opacity * self.get_mask_binary

    # ---- loss phụ L_m (công thức 3), có trọng số theo vùng chi tiết cao ----
    def mask_loss(self, detail_regions: Optional[DetailRegionSet] = None) -> torch.Tensor:
        m = torch.sigmoid(self._mask).squeeze(-1)  # (N,)
        if detail_regions is not None:
            with torch.no_grad():
                w = detail_regions.weight(self.get_xyz)
        else:
            w = torch.ones_like(m)
        return (w * m).mean()

    # ---- prune theo mask nhị phân, có bảo vệ cứng vùng chi tiết cao ----
    @torch.no_grad()
    def apply_mask_pruning(self, detail_regions: Optional[DetailRegionSet] = None,
                            hard_protect: bool = True) -> int:
        m = torch.sigmoid(self._mask).squeeze(-1)
        prune_mask = m <= self.mask_threshold
        if detail_regions is not None and hard_protect:
            protected = detail_regions.inside(self.get_xyz)
            prune_mask = prune_mask & (~protected)
        n_pruned = int(prune_mask.sum().item())
        if n_pruned > 0:
            # self.tmp_radii trong bản gốc CHỈ tồn tại tạm thời bên trong
            # densify_and_prune() (được set = None ngay sau khi hàm đó chạy
            # xong) — nhưng prune_points() gốc lại luôn cắt tensor này theo
            # valid_points_mask bất kể đang gọi từ đâu. Vì apply_mask_pruning
            # chạy ĐỘC LẬP với densify_and_prune (không đi kèm nhau), tmp_radii
            # đang là None -> cấp tạm 1 tensor rồi trả lại None như cũ để không
            # phá vỡ giả định của phần densify gốc.
            restore_none = self.tmp_radii is None
            if restore_none:
                self.tmp_radii = torch.zeros(self.get_xyz.shape[0], device="cuda")
            self.prune_points(prune_mask)
            if restore_none:
                self.tmp_radii = None
        return n_pruned

    # ---- các hàm densify/prune gốc: chỉ cần thêm việc cắt/nối thêm _mask
    #      cùng lúc với các thuộc tính khác, logic gradient/opacity giữ nguyên ----
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                               new_opacities, new_scaling, new_rotation,
                               new_tmp_radii, new_mask=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "mask": new_mask if new_mask is not None else
                     inverse_sigmoid(torch.full((new_xyz.shape[0], 1), self.mask_init, device="cuda"))}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        from utils.general_utils import build_rotation
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_mask = self._mask[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,
                                    new_scaling, new_rotation, new_tmp_radii, new_mask)

        prune_filter = torch.cat((selected_pts_mask,
                                   torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
                                    new_scaling, new_rotation, new_tmp_radii, new_mask)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
