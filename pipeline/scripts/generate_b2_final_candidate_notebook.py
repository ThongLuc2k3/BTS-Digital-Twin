#!/usr/bin/env python3
"""Sinh notebook "final candidate": gop fix EXPOSURE_COMP voi depth supervision
that trong 1 lan chay GPU duy nhat, do gap deadline nop bai.

Boi canh (xem WORKLOG.md 2026-07-26): run 2a/2b da loai LOW_VRAM_PROFILE la
nguyen nhan sap diem. Dieu tra tiep theo (so cfg_args that + phase-correlation
tren anh render that) chi ra nghi pham that: EXPOSURE_COMP=1 mac dinh
(03_train_3dgs.sh dong 23, --train_test_exp) duoc dung o moi run prepared da
thu (pilot cu, 2a, 2b) nhung baseline_ref (0.6731) lai dung train_test_exp=False.
render_round1_test_poses.py khong ap dung exposure compensation cho test pose
moi -> anh dung hinh hoc nhung sai tong mau toan cuc -> PSNR sap.

Vi gap deadline, KHONG chay rieng 1 lan "2c" (chi tat exposure, khong depth)
roi moi chay depth sau (ton them 1 chu ky GPU day ~2-2.5h). Notebook nay gop
lai: prepared source + true depth supervision (--depths, cung co che da dung
thanh cong o downloads/B2_done 1/2/2 safe.ipynb) + LOW_VRAM_PROFILE=0 +
EXPOSURE_COMP=0 + RESOLUTION=-1 -- khop moi setting cua baseline_ref, CONG
THEM depth supervision (muc tieu goc cua B2).

Rui ro da biet: neu run nay van sap diem, se KHONG tach bach duoc do depth
hay do nghi pham khac con sot -- nhung duoi ap luc deadline day la danh doi
hop ly (xem WORKLOG.md).

Output ghi vao chinh gs_model that (pipeline/work/hcm0031/gs_model), vi day
la ung vien nop bai chu khong phai run cach ly nua.
"""
import copy
import json
from pathlib import Path

ROOT = Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin")
BASE_NOTEBOOK = ROOT / "downloads" / "B2_done.ipynb"

DEPTH_PATCH_CELL = """# ==============================================
# CELL X - Patch train script for extra GS args (--depths)
# ==============================================
from pathlib import Path

project_dir = Path(WORKDIR) / 'project'
train_script = project_dir / 'pipeline' / 'scripts' / '03_train_3dgs.sh'
text = train_script.read_text(encoding='utf-8')

if 'TRAIN_EXTRA_ARGS_RAW="${TRAIN_EXTRA_ARGS_RAW:-}"' not in text:
    text = text.replace(
        'CHECKPOINT_ITERATIONS_OVERRIDE="${CHECKPOINT_ITERATIONS_OVERRIDE:-}"\\n',
        'CHECKPOINT_ITERATIONS_OVERRIDE="${CHECKPOINT_ITERATIONS_OVERRIDE:-}"\\n'
        'TRAIN_EXTRA_ARGS_RAW="${TRAIN_EXTRA_ARGS_RAW:-}"\\n',
        1,
    )

if 'read -r -a TRAIN_EXTRA_ARGS' not in text:
    text = text.replace(
        'EXTRA_ARGS=()\\n',
        'TRAIN_EXTRA_ARGS=()\\n'
        'if [[ -n "$TRAIN_EXTRA_ARGS_RAW" ]]; then\\n'
        '  read -r -a TRAIN_EXTRA_ARGS <<< "$TRAIN_EXTRA_ARGS_RAW"\\n'
        'fi\\n\\n'
        'EXTRA_ARGS=()\\n',
        1,
    )

if '"${TRAIN_EXTRA_ARGS[@]}" \\\\' not in text:
    text = text.replace(
        '  --resolution "$RESOLUTION" \\\\\\n'
        '  "${EXTRA_ARGS[@]}" \\\\\\n',
        '  --resolution "$RESOLUTION" \\\\\\n'
        '  "${TRAIN_EXTRA_ARGS[@]}" \\\\\\n'
        '  "${EXTRA_ARGS[@]}" \\\\\\n',
        1,
    )

train_script.write_text(text, encoding='utf-8')
print('Patched train script for TRAIN_EXTRA_ARGS_RAW:', train_script)
"""

CONFIG_CELL = """# =====================================================
# CELL X - Final candidate: prepared + depth + EXPOSURE_COMP=0 (khop baseline)
# =====================================================
from pathlib import Path

LOW_VRAM_PROFILE_OVERRIDE = "0"
RESOLUTION_OVERRIDE = "-1"
EXPOSURE_COMP_OVERRIDE = "0"

scene_root_check = PROJECT_DIR / 'pipeline' / 'work' / SCENE
depth_maps_dir = scene_root_check / 'colmap' / 'dense' / 'stereo' / 'depth_maps'
assert depth_maps_dir.exists() and any(depth_maps_dir.iterdir()), f'Khong thay depth maps: {depth_maps_dir} (can RUN_DENSE=1 chay truoc do)'

gs_repo_path = Path(GS_REPO)
support_files = [
    gs_repo_path / 'train.py',
    gs_repo_path / 'arguments' / '__init__.py',
    gs_repo_path / 'scene' / '__init__.py',
]
support_text = '\\n'.join(p.read_text(encoding='utf-8', errors='ignore') for p in support_files if p.exists())
assert 'depths' in support_text, 'GS_REPO hien tai khong ho tro --depths, dung lai truoc khi train.'

depths_arg = f'--depths {depth_maps_dir}'
print('=== Final candidate: prepared + depth + exposure_off ===')
print('LOW_VRAM_PROFILE override =', LOW_VRAM_PROFILE_OVERRIDE)
print('RESOLUTION override =', RESOLUTION_OVERRIDE)
print('EXPOSURE_COMP override =', EXPOSURE_COMP_OVERRIDE, '(khop baseline_ref: train_test_exp=False)')
print('TRAIN_EXTRA_ARGS_RAW =', depths_arg)
"""

REPORT_CELL = """# =====================================================
# CELL X - Final candidate: bao cao ket qua
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

print('=== Ket qua final candidate (full-image, n=%d) ===' % len(rows))
for k in fields:
    print(f'{k:>6} = {means[k]:.4f}')

score = means['score']
print()
print(f'So voi baseline (0.6731): delta = {score - BASELINE_SCORE:+.4f}')
if score >= BASELINE_SCORE:
    print('KET QUA: THANG baseline -> UNG VIEN NOP BAI TOT.')
elif score >= BASELINE_SCORE - 0.02:
    print('KET QUA: xap xi baseline -> co the dung tam, nhung khong ro rang thang.')
else:
    print('KET QUA: VAN duoi baseline dang ke -> con van de khac ngoai EXPOSURE_COMP/depth, can xem lai truoc khi nop.')
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
            "# BTS Digital Twin - Final Candidate - Depth + EXPOSURE_COMP=0 (`hcm0031`)",
        )
    )

    depth_patch_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_block(DEPTH_PATCH_CELL),
    }
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

    # Chen patch cell + config cell ngay truoc cell train/render/eval (index goc = 20)
    out["cells"].insert(20, depth_patch_cell)
    out["cells"].insert(21, config_cell)

    # Patch cell train/render/eval (shift +2 do vua chen 2 cell truoc no)
    train_cell_idx = 22
    src = "".join(out["cells"][train_cell_idx]["source"])
    old = "env['SOURCE_MODE'] = 'prepared'\nenv['ITERATION'] = '-1'\n"
    new = (
        old
        + "env['LOW_VRAM_PROFILE'] = LOW_VRAM_PROFILE_OVERRIDE\n"
        + "env['RESOLUTION'] = RESOLUTION_OVERRIDE\n"
        + "env['EXPOSURE_COMP'] = EXPOSURE_COMP_OVERRIDE\n"
        + "env['TRAIN_EXTRA_ARGS_RAW'] = depths_arg\n"
    )
    out["cells"][train_cell_idx]["source"] = to_source_block(replace_once(src, old, new))

    # Them report cell ngay sau cell train/render/eval
    out["cells"].insert(train_cell_idx + 1, report_cell)

    return out


def main() -> None:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    variant = build_variant(nb)
    out_path = ROOT / "downloads" / "B2_final_candidate_depth_exposure_off.ipynb"
    out_path.write_text(json.dumps(variant, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
