#!/usr/bin/env python3
"""Sinh 2 notebook cho Thi nghiem A: cach ly nguyen nhan sap diem cua B2.

Thiet ke 2-run bat buoc (xem .ai-debate/04_claude_final_review.md, xac nhan
lai trong .ai-debate/07_claude_audit.md muc 3):

- Run 2a (override tich cuc): source=prepared, KHONG depth, LOW_VRAM_PROFILE=0
  ep tuong minh, RESOLUTION=-1. Ky vong: score phuc hoi ve gan baseline 0.6731.
- Run 2b (control am): source=prepared, KHONG depth, LOW_VRAM_PROFILE=1
  ep tuong minh (tai lap dung cau hinh da gay sap diem truoc day).
  Ky vong: tai lap lai vung Score ~0.40.

Chi khi CA HAI dieu kien tren cung dung, moi du bang chung nhan qua de
ket luan LOW_VRAM_PROFILE la bien quyet dinh (khong phai ban chat
prepared/depth). Hai run deu KHONG dung --depths de khong bi confound voi
cau hoi depth supervision.

Output cua ca hai run duoc redirect sang pipeline/work/hcm0031/trick_runs/
de KHONG ghi de len gs_model that (dang la baseline chua day du artifact).
"""
import copy
import json
from pathlib import Path

ROOT = Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin")
BASE_NOTEBOOK = ROOT / "downloads" / "B2_done.ipynb"

CONFIG_CELL_TEMPLATE = """# =====================================================
# CELL X - Thi nghiem A: cau hinh isolation run {run_id}
# =====================================================
from pathlib import Path

ISOLATION_RUN_ID = "{run_id}"
LOW_VRAM_PROFILE_OVERRIDE = "{low_vram_profile}"
RESOLUTION_OVERRIDE = "-1"

isolation_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE / 'trick_runs' / f'b2_isolation_{{ISOLATION_RUN_ID}}'
isolation_model_dir = isolation_root / 'gs_model'
isolation_render_dir = isolation_root / 'renders'
isolation_eval_csv = isolation_root / 'eval_metrics.csv'
isolation_root.mkdir(parents=True, exist_ok=True)

print('=== Thi nghiem A - isolation run:', ISOLATION_RUN_ID, '===')
print('LOW_VRAM_PROFILE override =', LOW_VRAM_PROFILE_OVERRIDE)
print('RESOLUTION override =', RESOLUTION_OVERRIDE)
print('KHONG dung --depths trong run nay (isolate rieng bien LOW_VRAM_PROFILE)')
print('isolation_model_dir =', isolation_model_dir)
print('isolation_render_dir =', isolation_render_dir)
print('isolation_eval_csv =', isolation_eval_csv)
"""

REPORT_CELL_TEMPLATE = """# =====================================================
# CELL X - Thi nghiem A: bao cao ket qua run {run_id}
# =====================================================
import csv
from statistics import mean

BASELINE_SCORE = 0.6731
FAILED_RUN_SCORE_RANGE = (0.40, 0.41)  # tu downloads/B2_done.ipynb va B2_done 2 safe.ipynb

with open(isolation_eval_csv, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
assert rows, f'eval csv rong: {{isolation_eval_csv}}'

fields = ('psnr', 'ssim', 'lpips', 'score')
means = {{k: mean(float(r[k]) for r in rows) for k in fields}}

print('=== Ket qua run {run_id} (full-image, n=%d) ===' % len(rows))
for k in fields:
    print(f'{{k:>6}} = {{means[k]:.4f}}')

score = means['score']
print()
print(f'So voi baseline (0.6731): delta = {{score - BASELINE_SCORE:+.4f}}')
print(f'So voi failed run cu (~0.40-0.41): delta = {{score - FAILED_RUN_SCORE_RANGE[0]:+.4f}}')

if '{run_id}' == '2a_low_vram_off':
    if score >= BASELINE_SCORE - 0.02:
        print('KET LUAN 2a: PHUC HOI ve gan baseline -> LOW_VRAM_PROFILE la nghi phan hop ly.')
    elif FAILED_RUN_SCORE_RANGE[0] - 0.05 <= score <= FAILED_RUN_SCORE_RANGE[1] + 0.05:
        print('KET LUAN 2a: VAN SAP diem nhu cu -> LOW_VRAM_PROFILE KHONG phai nguyen nhan chinh,')
        print('loi nam o ban than prepared source hoac noi khac. Dieu tra tiep source/parity.')
    else:
        print('KET LUAN 2a: ket qua nam ngoai ca 2 vung du doan, can xem log train truoc khi ket luan.')
else:
    if FAILED_RUN_SCORE_RANGE[0] - 0.05 <= score <= FAILED_RUN_SCORE_RANGE[1] + 0.05:
        print('KET LUAN 2b: TAI LAP duoc vung sap diem ~0.40 -> xac nhan cau hinh nay that su gay loi,')
        print('cung co nhan qua cho ket luan cua run 2a.')
    else:
        print('KET LUAN 2b: KHONG tai lap duoc vung sap diem cu -> failed run truoc co the')
        print('do nguyen nhan khac (vd. state random, phien ban repo khac o thoi diem do), can than trong')
        print('khi dien giai run 2a.')

print()
print('CHI ket luan LOW_VRAM_PROFILE la nguyen nhan neu CA HAI run 2a va 2b deu khop du doan.')
"""


def to_source_block(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Khong tim thay doan can thay: {old[:80]!r}")
    return text.replace(old, new, 1)


def build_variant(nb: dict, *, run_id: str, low_vram_profile: str, title_suffix: str) -> dict:
    out = copy.deepcopy(nb)

    cell0 = "".join(out["cells"][0]["source"])
    out["cells"][0]["source"] = to_source_block(
        cell0.replace(
            "# BTS Digital Twin - B2 Pilot (`hcm0031`)",
            f"# BTS Digital Twin - {title_suffix} (`hcm0031`)",
        )
    )

    config_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(
            CONFIG_CELL_TEMPLATE.format(run_id=run_id, low_vram_profile=low_vram_profile)
        ),
    }
    report_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(REPORT_CELL_TEMPLATE.format(run_id=run_id)),
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

    variant_2a = build_variant(
        nb,
        run_id="2a_low_vram_off",
        low_vram_profile="0",
        title_suffix="B2 Isolation 2a - LOW_VRAM_PROFILE=0 (override)",
    )
    variant_2b = build_variant(
        nb,
        run_id="2b_low_vram_control",
        low_vram_profile="1",
        title_suffix="B2 Isolation 2b - LOW_VRAM_PROFILE=1 (control am)",
    )

    out_2a = ROOT / "downloads" / "B2_isolation_2a_low_vram_off.ipynb"
    out_2b = ROOT / "downloads" / "B2_isolation_2b_low_vram_control.ipynb"
    out_2a.write_text(json.dumps(variant_2a, ensure_ascii=False, indent=1), encoding="utf-8")
    out_2b.write_text(json.dumps(variant_2b, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_2a)
    print(out_2b)


if __name__ == "__main__":
    main()
