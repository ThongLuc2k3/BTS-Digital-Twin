"""Liệt kê scene trong Dataset/VAI_NVS_DATA/phase1 và đường dẫn con của từng scene.

Tên scene lấy đúng theo tên thư mục thật trên đĩa (đã kiểm tra thủ công, xem
Dataset/README.md) — KHÔNG dùng tên minh hoạ "scene_001" trong đề bài.
Lưu ý 2 scene public dùng chữ thường (hcm0031, hcm0034), phân biệt hoa/thường trên Linux.

Đường dẫn dataset ưu tiên lấy từ biến môi trường BTS_DATASET_ROOT nếu có — cần
thiết khi code và dataset KHÔNG nằm chung 1 thư mục gốc (vd trên Kaggle: code lấy
từ git clone, dataset tải riêng từ Google Drive, 2 nơi khác gốc nhau nên không thể
suy ra bằng "đi lên N cấp từ vị trí file code" như lúc chạy local).
"""
import os
from dataclasses import dataclass
from pathlib import Path

_env_root = os.environ.get("BTS_DATASET_ROOT")
if _env_root:
    DATASET_ROOT = Path(_env_root)
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATASET_ROOT = REPO_ROOT / "Dataset" / "VAI_NVS_DATA" / "phase1"

PUBLIC_SCENES = ["HCM0181", "HCM0193", "HCM0204", "hcm0031", "hcm0034"]
PRIVATE_SCENES = [
    "HCM0249", "HCM0254", "HCM0276", "HCM1439",
    "HNI0131", "HNI0265", "HNI0366", "HNI0437",
]


@dataclass
class Scene:
    name: str
    split: str  # "public" | "private"
    root: Path

    @property
    def train_images_dir(self) -> Path:
        return self.root / "train" / "images"

    @property
    def provided_sparse_dir(self) -> Path:
        """sparse/0 do BTC cung cấp sẵn — chỉ HCM0249 là có dữ liệu thật, xem Dataset/README.md."""
        return self.root / "train" / "sparse" / "0"

    @property
    def test_poses_csv(self) -> Path:
        return self.root / "test" / "test_poses.csv"

    @property
    def gt_test_images_dir(self) -> Path:
        """Chỉ tồn tại ở public_set — ảnh thật dùng để tự chấm PSNR/SSIM/LPIPS."""
        return self.root / "test" / "images"

    def has_valid_provided_sparse(self) -> bool:
        d = self.provided_sparse_dir
        if not d.exists():
            return False
        cams = d / "cameras.bin"
        return cams.exists() and cams.stat().st_size > 0


def get_scene(name: str) -> Scene:
    if name in PUBLIC_SCENES:
        return Scene(name=name, split="public", root=DATASET_ROOT / "public_set" / name)
    if name in PRIVATE_SCENES:
        return Scene(name=name, split="private", root=DATASET_ROOT / "private_set1" / name)
    raise ValueError(
        f"Không rõ scene '{name}'. Public: {PUBLIC_SCENES}. Private: {PRIVATE_SCENES}."
    )


def all_scenes(split: str | None = None) -> list[Scene]:
    """split=None -> toàn bộ 13 scene; split="public"/"private" -> chỉ tập tương ứng."""
    names: list[str] = []
    if split in (None, "public"):
        names += PUBLIC_SCENES
    if split in (None, "private"):
        names += PRIVATE_SCENES
    return [get_scene(n) for n in names]
