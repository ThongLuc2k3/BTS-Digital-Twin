#!/usr/bin/env python3
"""Sinh notebook cho Thi nghiem B: cach ly EXPOSURE_COMP (--train_test_exp).

Boi canh (xem WORKLOG.md muc 2026-07-26): run 2a/2b (Thi nghiem A) da loai
LOW_VRAM_PROFILE la nguyen nhan (2a tat het van sap gan bang 2b). Dieu tra
sau do (so cfg_args that + phase-correlation tren anh render that) chi ra
nghi phan that: baseline_ref (0.6731) dung train_test_exp=False, nhung moi
run prepared (pilot cu, 2a, 2b) deu de EXPOSURE_COMP=1 mac dinh (03_train_3dgs.sh
dong 23) -> bat --train_test_exp. render_round1_test_poses.py khong ap dung
exposure compensation cho test pose moi -> anh dung hinh hoc nhung sai tong
mau toan cuc -> PSNR sap trong khi SSIM/LPIPS chi giam vua phai.

Run 2c: prepared, KHONG depth, LOW_VRAM_PROFILE=0, RESOLUTION=-1,
EXPOSURE_COMP=0 -- khop CHINH XAC cau hinh baseline_ref, chi khac nguon
dense/depth. Neu score phuc hoi gan 0.6731 -> xac nhan EXPOSURE_COMP la
nguyen nhan that, mo lai nhanh prepared + true depths.
"""
import copy
import json
from pathlib import Path

ROOT = Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin")
BASE_NOTEBOOK = ROOT / "downloads" / "B2_done.ipynb"

RUN_ID = "2c_exposure_off"

CONFIG_CELL = """# =====================================================
# CELL X - Thi nghiem B: cau hinh isolation run 2c_exposure_off
# =====================================================
from pathlib import Path

ISOLATION_RUN_ID = "2c_exposure_off"
LOW_VRAM_PROFILE_OVERRIDE = "0"
RESOLUTION_OVERRIDE = "-1"
EXPOSURE_COMP_OVERRIDE = "0"

isolation_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE / 'trick_runs' / f'b2_isolation_{ISOLATION_RUN_ID}'
isolation_model_dir = isolation_root / 'gs_model'
isolation_render_dir = isolation_root / 'renders'
isolation_eval_csv = isolation_root / 'eval_metrics.csv'
isolation_root.mkdir(parents=True, exist_ok=True)

print('=== Thi nghiem B - isolation run:', ISOLATION_RUN_ID, '===')
print('LOW_VRAM_PROFILE override =', LOW_VRAM_PROFILE_OVERRIDE)
print('RESOLUTION override =', RESOLUTION_OVERRIDE)
print('EXPOSURE_COMP override =', EXPOSURE_COMP_OVERRIDE, '(khop baseline_ref: train_test_exp=False)')
print('KHONG dung --depths trong run nay (isolate rieng bien EXPOSURE_COMP)')
print('isolation_model_dir =', isolation_model_dir)
print('isolation_render_dir =', isolation_render_dir)
print('isolation_eval_csv =', isolation_eval_csv)
"""

REPORT_CELL = """# =====================================================
# CELL X - Thi nghiem B: bao cao ket qua run 2c_exposure_off
# =====================================================
import csv
from statistics import mean

BASELINE_SCORE = 0.6731
FAILED_RUN_SCORE_RANGE = (0.40, 0.49)  # vung sap diem quan sat o pilot cu + 2a/2b

with open(isolation_eval_csv, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
assert rows, f'eval csv rong: {isolation_eval_csv}'

fields = ('psnr', 'ssim', 'lpips', 'score')
means = {k: mean(float(r[k]) for r in rows) for k in fields}

print('=== Ket qua run 2c_exposure_off (full-image, n=%d) ===' % len(rows))
for k in fields:
    print(f'{k:>6} = {means[k]:.4f}')

score = means['score']
print()
print(f'So voi baseline (0.6731): delta = {score - BASELINE_SCORE:+.4f}')
print(f'So voi vung sap diem cu (~0.40-0.49): delta = {score - FAILED_RUN_SCORE_RANGE[0]:+.4f}')

if score >= BASELINE_SCORE - 0.02:
    print('KET LUAN 2c: PHUC HOI ve gan baseline -> XAC NHAN EXPOSURE_COMP/train_test_exp la nguyen nhan that.')
    print('-> Mo lai nhanh prepared + true depths (muc tieu goc cua B2).')
elif FAILED_RUN_SCORE_RANGE[0] - 0.05 <= score <= FAILED_RUN_SCORE_RANGE[1] + 0.05:
    print('KET LUAN 2c: VAN sap diem nhu 2a/2b -> EXPOSURE_COMP KHONG phai nguyen nhan duy nhat.')
    print('Con nghi pham khac chua lo dien, can dieu tra tiep truoc khi thu depth.')
else:
    print('KET LUAN 2c: ket qua nam ngoai ca 2 vung du doan, can xem log train va anh render truoc khi ket luan.')
"""


def to_source_block(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Khong tim thay doan can thay: {old[:80]!r}")
    return text.replace(old, new, 1)


def build_variant(nb: dict) -> dict:
    out = copy.deepcopy(nb)

    cell0 = "".join(out["cells"][0]["source"])
    out["cells"][0]["source"] = to_source_block(
        cell0.replace(
            "# BTS Digital Twin - B2 Pilot (`hcm0031`)",
            "# BTS Digital Twin - B2 Isolation 2c - EXPOSURE_COMP=0 (khop baseline) (`hcm0031`)",
        )
    )

    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(CONFIG_CELL),
    }
    report_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(REPORT_CELL),
    }

    # Chen config cell ngay truoc cell train/render/eval (index goc = 20)
    out["cells"].insert(20, config_cell)

    # Patch cell train/render/eval (shift +1 do vua chen 1 cell truoc no)
    train_cell_idx = 21
    src = "".join(out["cells"][train_cell_idx]["source"])
    old = "env['SOURCE_MODE'] = 'prepared'\nenv['ITERATION'] = '-1'\n"
    new = (
        old
        + "env['LOW_VRAM_PROFILE'] = LOW_VRAM_PROFILE_OVERRIDE\n"
        + "env['RESOLUTION'] = RESOLUTION_OVERRIDE\n"
        + "env['EXPOSURE_COMP'] = EXPOSURE_COMP_OVERRIDE\n"
        + "env['MODEL_DIR'] = str(isolation_model_dir)\n"
        + "env['RENDER_OUT_DIR'] = str(isolation_render_dir)\n"
        + "env['EVAL_OUT_CSV'] = str(isolation_eval_csv)\n"
    )
    out["cells"][train_cell_idx]["source"] = to_source_block(replace_once(src, old, new))

    # Them report cell ngay sau cell train/render/eval
    out["cells"].insert(train_cell_idx + 1, report_cell)

    return out


def main() -> None:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    variant = build_variant(nb)
    out_path = ROOT / "downloads" / "B2_isolation_2c_exposure_off.ipynb"
    out_path.write_text(json.dumps(variant, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
