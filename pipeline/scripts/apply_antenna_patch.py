#!/usr/bin/env python3
"""Vá `train.py` của repo GỐC graphdeco-inria/gaussian-splatting (vừa clone ở
Bước 2 của notebook) để hỗ trợ 2 ý tưởng "tập trung train vào 1 vùng nhỏ"
(vd ăn-ten) — không rewrite lại cả trainer, chỉ thêm đúng vài dòng ở 2 chỗ:

  1. Trọng số hoá loss theo vùng (loss masking) — vùng trong file
     antenna_weights.json (do 07_build_antenna_weights.py tạo) được nhân L1
     loss cao hơn (SSIM giữ nguyên, không mask — SSIM tính theo cửa sổ trượt,
     mask hoá phức tạp hơn nhiều so với lợi ích thêm được).
  2. Chọn lại tần suất camera (view resampling) — ảnh có "coverage" (diện tích
     vùng đó chiếm trong ảnh) cao được train.py chọn thường xuyên hơn ảnh khác,
     thay vì chọn đều mọi ảnh như mặc định.

Chỉ áp dụng khi chạy train.py kèm --antenna_weights_json <path> — không dùng cờ
này thì train.py chạy y hệt bản gốc (patch tự return False, không đổi hành vi).

Cách dùng (chạy 1 lần ngay sau khi clone gaussian-splatting, trước khi train):
    python apply_antenna_patch.py --gs_repo /kaggle/working/gaussian-splatting

An toàn: mỗi đoạn vá đối chiếu ĐÚNG 1 lần với text gốc đã lưu sẵn (fetch trực
tiếp từ graphdeco-inria/gaussian-splatting nhánh main để viết script này) —
nếu không khớp (repo gốc đổi code, hoặc file đã bị vá trước đó) sẽ báo lỗi rõ
ràng và dừng lại, KHÔNG âm thầm bỏ qua.
"""
import argparse
import sys
from pathlib import Path

PATCHES: list[tuple[str, str, str]] = [
    (
        "import json",
        "import os\nimport torch\n",
        "import os\nimport json\nimport torch\n",
    ),
    (
        "import random.choices",
        "from random import randint\n",
        "from random import randint, choices\n",
    ),
    (
        "argparse: --antenna_weights_json/--antenna_weight",
        '    parser.add_argument("--start_checkpoint", type=str, default = None)\n'
        '    args = parser.parse_args(sys.argv[1:])',
        '    parser.add_argument("--start_checkpoint", type=str, default = None)\n'
        '    parser.add_argument("--antenna_weights_json", type=str, default=None,\n'
        '                        help="pipeline/work/<scene>/antenna_weights.json — bat tap trung loss + resample view vao 1 vung nho (vd anten)")\n'
        '    parser.add_argument("--antenna_weight", type=float, default=None,\n'
        '                        help="Ghi de he so nhan loss trong vung do (mac dinh: lay tu file json)")\n'
        '    args = parser.parse_args(sys.argv[1:])',
    ),
    (
        "training(): them 2 tham so antenna_weights_json/antenna_weight_override",
        "def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):",
        "def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, antenna_weights_json=None, antenna_weight_override=None):",
    ),
    (
        "call site: truyen antenna_weights_json/antenna_weight",
        "    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)",
        "    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.antenna_weights_json, args.antenna_weight)",
    ),
    (
        "load antenna_weights.json + ham dung mask",
        "    viewpoint_stack = scene.getTrainCameras().copy()\n"
        "    viewpoint_indices = list(range(len(viewpoint_stack)))\n"
        "    ema_loss_for_log = 0.0\n"
        "    ema_Ll1depth_for_log = 0.0",
        "    viewpoint_stack = scene.getTrainCameras().copy()\n"
        "    viewpoint_indices = list(range(len(viewpoint_stack)))\n"
        "    ema_loss_for_log = 0.0\n"
        "    ema_Ll1depth_for_log = 0.0\n"
        "\n"
        "    # --- antenna-focus (xem pipeline/scripts/07_build_antenna_weights.py) ---\n"
        "    antenna_boxes = {}\n"
        "    antenna_weight_value = 4.0\n"
        "    antenna_cam_weights = None\n"
        "    all_train_cams = None\n"
        "    if antenna_weights_json:\n"
        "        with open(antenna_weights_json) as _af:\n"
        "            _antenna_data = json.load(_af)\n"
        "        antenna_boxes = _antenna_data.get(\"images\", {})\n"
        "        antenna_weight_value = antenna_weight_override if antenna_weight_override is not None else _antenna_data.get(\"weight_value\", 4.0)\n"
        "        all_train_cams = scene.getTrainCameras().copy()\n"
        "        antenna_cam_weights = [max(antenna_boxes.get(cam.image_name, {}).get(\"coverage\", 0.0), 0.02) for cam in all_train_cams]\n"
        "        n_with_box = sum(1 for c in all_train_cams if c.image_name in antenna_boxes)\n"
        "        print(f\"[antenna-focus] {antenna_weights_json}: {n_with_box}/{len(all_train_cams)} anh co vung duoc danh dau, weight={antenna_weight_value}\")\n"
        "\n"
        "    def _antenna_mask(cam):\n"
        "        box = antenna_boxes.get(cam.image_name, {}).get(\"box\")\n"
        "        if box is None:\n"
        "            return None\n"
        "        x0, y0, x1, y1 = box\n"
        "        mask = torch.ones((1, cam.image_height, cam.image_width), device=\"cuda\")\n"
        "        x0i, x1i = max(int(x0), 0), min(int(x1) + 1, cam.image_width)\n"
        "        y0i, y1i = max(int(y0), 0), min(int(y1) + 1, cam.image_height)\n"
        "        mask[:, y0i:y1i, x0i:x1i] = antenna_weight_value\n"
        "        return mask",
    ),
    (
        "pick camera: weighted resampling khi bat antenna-focus",
        "        # Pick a random Camera\n"
        "        if not viewpoint_stack:\n"
        "            viewpoint_stack = scene.getTrainCameras().copy()\n"
        "            viewpoint_indices = list(range(len(viewpoint_stack)))\n"
        "        rand_idx = randint(0, len(viewpoint_indices) - 1)\n"
        "        viewpoint_cam = viewpoint_stack.pop(rand_idx)\n"
        "        vind = viewpoint_indices.pop(rand_idx)",
        "        # Pick a random Camera\n"
        "        if antenna_cam_weights is not None:\n"
        "            viewpoint_cam = choices(all_train_cams, weights=antenna_cam_weights, k=1)[0]\n"
        "        else:\n"
        "            if not viewpoint_stack:\n"
        "                viewpoint_stack = scene.getTrainCameras().copy()\n"
        "                viewpoint_indices = list(range(len(viewpoint_stack)))\n"
        "            rand_idx = randint(0, len(viewpoint_indices) - 1)\n"
        "            viewpoint_cam = viewpoint_stack.pop(rand_idx)\n"
        "            vind = viewpoint_indices.pop(rand_idx)",
    ),
    (
        "loss: weighted L1 trong vung duoc danh dau",
        "        # Loss\n"
        "        gt_image = viewpoint_cam.original_image.cuda()\n"
        "        Ll1 = l1_loss(image, gt_image)\n"
        "        if FUSED_SSIM_AVAILABLE:",
        "        # Loss\n"
        "        gt_image = viewpoint_cam.original_image.cuda()\n"
        "        _antenna_w = _antenna_mask(viewpoint_cam)\n"
        "        if _antenna_w is not None:\n"
        "            Ll1 = (_antenna_w * torch.abs(image - gt_image)).sum() / _antenna_w.expand_as(image).sum()\n"
        "        else:\n"
        "            Ll1 = l1_loss(image, gt_image)\n"
        "        if FUSED_SSIM_AVAILABLE:",
    ),
]


def apply_patches(train_py: Path) -> None:
    content = train_py.read_text(encoding="utf-8")

    if "antenna-focus" in content:
        print(f"-> {train_py} đã được vá antenna-focus từ trước, bỏ qua (không vá lại 2 lần).")
        return

    for label, old, new in PATCHES:
        n = content.count(old)
        if n != 1:
            raise SystemExit(
                f"[LỖI] Vá '{label}' thất bại: tìm thấy {n} lần đoạn text cần thay (cần đúng 1).\n"
                f"Khả năng cao graphdeco-inria/gaussian-splatting đã đổi code kể từ lúc viết patch này "
                f"— cần đối chiếu lại {train_py} thủ công, KHÔNG bỏ qua bước này."
            )
        content = content.replace(old, new, 1)

    train_py.write_text(content, encoding="utf-8")
    print(f"-> Đã vá xong {train_py} ({len(PATCHES)} chỗ) — chỉ có tác dụng khi train.py chạy kèm "
          f"--antenna_weights_json <path>, không có cờ đó thì hành vi y hệt bản gốc.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_repo", required=True, help="Thư mục đã clone --recursive graphdeco-inria/gaussian-splatting")
    args = ap.parse_args()

    train_py = Path(args.gs_repo) / "train.py"
    if not train_py.exists():
        raise SystemExit(f"Không thấy {train_py} — kiểm tra lại --gs_repo.")
    apply_patches(train_py)


if __name__ == "__main__":
    main()
