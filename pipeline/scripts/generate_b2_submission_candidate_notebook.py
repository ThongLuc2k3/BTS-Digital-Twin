#!/usr/bin/env python3
"""Sinh notebook "submission candidate": EXPOSURE_COMP=0 (fix da xac nhan) +
safe-densify overrides (chan OOM vua gap o run 2c tren Colab T4) + ghi thang
vao gs_model that (khong phai trick_runs) de dung ngay lam bai nop.

Boi canh (xem WORKLOG.md 2026-07-26):
- 2c (LOW_VRAM_PROFILE=0 ep tat hoan toan, khop moi setting baseline) bi
  torch.OutOfMemoryError luc iteration 11200/30000 tren Colab T4 (14.56GiB),
  vi khong gioi han toc do densify.
- LOW_VRAM_PROFILE da duoc loai la nguyen nhan gay sap diem that (2a tat het
  van sap vi EXPOSURE_COMP) -> bat lai gioi han densify (KHONG bat toan bo
  LOW_VRAM_PROFILE=1, vi cai do keo RESOLUTION xuong 4, lech baseline) la an
  toan, khong keo lai bug that.
- baseline_ref (0.6731) tu no cung KHONG dung depth -> khong dung --depths o
  day khong lam diem thap hon baseline; depth van dang hong (xem worklog),
  de lai lam viec sau neu con thoi gian.

Run nay: prepared + EXPOSURE_COMP=0 + RESOLUTION=-1 + densify throttle (chan
OOM) + PYTORCH_CUDA_ALLOC_CONF, ghi thang vao gs_model/renders_b2_pilot/
eval_metrics_b2_pilot.csv that (khong redirect trick_runs) vi day la ung vien
nop bai, khong phai run cach ly.
"""
import copy
import json
from pathlib import Path

ROOT = Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin")
BASE_NOTEBOOK = ROOT / "downloads" / "B2_done.ipynb"

CONFIG_CELL = """# =====================================================
# CELL X - Submission candidate: EXPOSURE_COMP=0 + safe-densify (chan OOM)
# =====================================================
LOW_VRAM_PROFILE_OVERRIDE = "0"
RESOLUTION_OVERRIDE = "-1"
EXPOSURE_COMP_OVERRIDE = "0"
DENSIFY_UNTIL_ITER_OVERRIDE = "15000"
DENSIFY_FROM_ITER_OVERRIDE = "500"
DENSIFICATION_INTERVAL_OVERRIDE = "100"
PERCENT_DENSE_OVERRIDE = "0.01"
OPACITY_RESET_INTERVAL_OVERRIDE = "3000"
PYTORCH_CUDA_ALLOC_CONF_OVERRIDE = "expandable_segments:True"

print('=== Submission candidate: prepared + exposure_off + safe-densify ===')
print('LOW_VRAM_PROFILE override =', LOW_VRAM_PROFILE_OVERRIDE, '(giu 0 de KHONG bi ep RESOLUTION xuong 4)')
print('RESOLUTION override =', RESOLUTION_OVERRIDE)
print('EXPOSURE_COMP override =', EXPOSURE_COMP_OVERRIDE, '(khop baseline_ref: train_test_exp=False)')
print('DENSIFY_UNTIL_ITER =', DENSIFY_UNTIL_ITER_OVERRIDE, '(chan OOM gap o run 2c)')
print('KHONG dung --depths trong run nay (dang hong, xem WORKLOG.md)')
print('Ghi thang vao gs_model/renders_b2_pilot/eval_metrics_b2_pilot.csv THAT (khong phai trick_runs)')
"""

REPORT_CELL = """# =====================================================
# CELL X - Submission candidate: bao cao ket qua
# =====================================================
import csv
from statistics import mean

BASELINE_SCORE = 0.6731

eval_csv = PROJECT_DIR / 'pipeline' / 'work' / SCENE / 'eval_metrics_b2_pilot.csv'
with open(eval_csv, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
assert rows, f'eval csv rong: {eval_csv}'

fields = ('psnr', 'ssim', 'lpips', 'score')
means = {k: mean(float(r[k]) for r in rows) for k in fields}

print('=== Ket qua submission candidate (full-image, n=%d) ===' % len(rows))
for k in fields:
    print(f'{k:>6} = {means[k]:.4f}')

score = means['score']
print()
print(f'So voi baseline (0.6731): delta = {score - BASELINE_SCORE:+.4f}')
if score >= BASELINE_SCORE:
    print('KET QUA: THANG baseline -> UNG VIEN NOP BAI TOT.')
elif score >= BASELINE_SCORE - 0.02:
    print('KET QUA: xap xi baseline -> co the dung tam.')
else:
    print('KET QUA: van duoi baseline dang ke -> con nghi pham khac, xem lai truoc khi nop.')
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
            "# BTS Digital Twin - Submission Candidate - EXPOSURE_COMP=0 + safe-densify (`hcm0031`)",
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
        + "env['DENSIFY_UNTIL_ITER'] = DENSIFY_UNTIL_ITER_OVERRIDE\n"
        + "env['DENSIFY_FROM_ITER'] = DENSIFY_FROM_ITER_OVERRIDE\n"
        + "env['DENSIFICATION_INTERVAL'] = DENSIFICATION_INTERVAL_OVERRIDE\n"
        + "env['PERCENT_DENSE'] = PERCENT_DENSE_OVERRIDE\n"
        + "env['OPACITY_RESET_INTERVAL'] = OPACITY_RESET_INTERVAL_OVERRIDE\n"
        + "env['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF_OVERRIDE\n"
    )
    out["cells"][train_cell_idx]["source"] = to_source_block(replace_once(src, old, new))

    # Them report cell ngay sau cell train/render/eval
    out["cells"].insert(train_cell_idx + 1, report_cell)

    return out


def main() -> None:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    variant = build_variant(nb)
    out_path = ROOT / "downloads" / "B2_submission_candidate_exposure_off_safe_densify.ipynb"
    out_path.write_text(json.dumps(variant, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
