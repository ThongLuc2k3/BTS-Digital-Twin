#!/usr/bin/env python3
"""Sinh notebook refine 2-stage theo flow:

1. Stage A: train full-scene 30k (config an toan da xac nhan).
2. Proxy no-GT: cham do sac net/nhieu tren render output, KHONG dung GT.
3. Stage B loop: resume checkpoint A, refine lap lai cac anh proxy < threshold.
4. Final GT eval: chi cham THAT o cuoi bang psnr_max=50.0.

Notebook nay van giu huong antenna-focus, nhung da them refine-image loop de
dung cho ca antenna/noise ma khong duoc phep dua vao GT de chon anh xau.
"""
import copy
import json
from pathlib import Path

ROOT = Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin")
BASE_NOTEBOOK = ROOT / "downloads" / "B2_done.ipynb"


def read_local(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


EMBEDDED_PROJECT_FILES = {
    "pipeline/scripts/03_train_3dgs.sh": read_local("pipeline/scripts/03_train_3dgs.sh"),
    "pipeline/scripts/04_run_colmap_dense.sh": read_local("pipeline/scripts/04_run_colmap_dense.sh"),
    "pipeline/scripts/eval_round1_metrics.py": read_local("pipeline/scripts/eval_round1_metrics.py"),
    "pipeline/scripts/build_tower_train_masks.py": read_local("pipeline/scripts/build_tower_train_masks.py"),
    "pipeline/scripts/colmap_read_model.py": read_local("pipeline/scripts/colmap_read_model.py"),
    "pipeline/scripts/score_refine_candidates_no_gt.py": read_local("pipeline/scripts/score_refine_candidates_no_gt.py"),
    "pipeline/scripts/map_refine_targets_to_train_views.py": read_local("pipeline/scripts/map_refine_targets_to_train_views.py"),
    "trick/scripts/bootstrap_tower_masks.py": read_local("trick/scripts/bootstrap_tower_masks.py"),
}

TOWER_BBOX_JSON_TEXT = read_local("pipeline/work/hcm0031/tower_bbox3d.json")

BOOTSTRAP_PROJECT_CELL = (
    "# =====================================================\n"
    "# CELL X - Seed patched project scripts into cloned repo\n"
    "# =====================================================\n"
    "from pathlib import Path\n"
    "import os\n"
    "\n"
    f"EMBEDDED_PROJECT_FILES = {json.dumps(EMBEDDED_PROJECT_FILES, ensure_ascii=False, indent=2)}\n"
    f"TOWER_BBOX_JSON_TEXT = {json.dumps(TOWER_BBOX_JSON_TEXT, ensure_ascii=False)}\n"
    "\n"
    "project_dir = Path(WORKDIR) / 'project'\n"
    "assert project_dir.exists(), f'Khong thay project dir sau clone: {project_dir}'\n"
    "\n"
    "for rel_path, content in EMBEDDED_PROJECT_FILES.items():\n"
    "    out_path = project_dir / rel_path\n"
    "    out_path.parent.mkdir(parents=True, exist_ok=True)\n"
    "    out_path.write_text(content, encoding='utf-8')\n"
    "    if out_path.suffix == '.sh':\n"
    "        os.chmod(out_path, 0o755)\n"
    "    print('Seeded:', out_path)\n"
    "\n"
    "bbox_path = project_dir / 'pipeline' / 'work' / 'hcm0031' / 'tower_bbox3d.json'\n"
    "bbox_path.parent.mkdir(parents=True, exist_ok=True)\n"
    "bbox_path.write_text(TOWER_BBOX_JSON_TEXT, encoding='utf-8')\n"
    "print('Seeded:', bbox_path)\n"
)

TRAIN_PY_PATCH_CELL = """# =====================================================
# CELL X - Patch train.py: them refine-image loss + antenna mask loss
# =====================================================
train_py = GS_REPO / 'train.py'
text = train_py.read_text(encoding='utf-8')

anchor = (
    "        else:\\n"
    "            Ll1depth = 0\\n"
    "\\n"
    "        loss.backward()\\n"
)
fallback_anchor = "        loss.backward()\\n"
patch = (
    "        else:\\n"
    "            Ll1depth = 0\\n"
    "\\n"
    "        refine_image_loss_weight = float(os.environ.get('REFINE_IMAGE_LOSS_WEIGHT', '0') or 0)\\n"
    "        if refine_image_loss_weight > 0:\\n"
    "            if 'REFINE_IMAGE_NAME_SET' not in globals():\\n"
    "                _refine_csv = os.environ.get('REFINE_IMAGE_LIST', '')\\n"
    "                _refine_names = set()\\n"
    "                if _refine_csv and os.path.isfile(_refine_csv):\\n"
    "                    import csv as _csv\\n"
    "                    with open(_refine_csv, newline='', encoding='utf-8') as _f:\\n"
    "                        for _row in _csv.DictReader(_f):\\n"
    "                            if int(float(_row.get('refine_flag', '0') or 0)) == 1:\\n"
    "                                _refine_names.add(_row['image'])\\n"
    "                globals()['REFINE_IMAGE_NAME_SET'] = _refine_names\\n"
    "            _refine_names = globals()['REFINE_IMAGE_NAME_SET']\\n"
    "            if viewpoint_cam.image_name in _refine_names:\\n"
    "                _refine_ll1 = torch.abs(image - gt_image).mean()\\n"
    "                loss = loss + refine_image_loss_weight * _refine_ll1\\n"
    "\\n"
    "        antenna_loss_weight = float(os.environ.get('ANTENNA_LOSS_WEIGHT', '0') or 0)\\n"
    "        if antenna_loss_weight > 0:\\n"
    "            antenna_mask_dir = os.environ.get('ANTENNA_MASK_DIR', '')\\n"
    "            if antenna_mask_dir:\\n"
    "                if 'ANTENNA_MASK_CACHE' not in globals():\\n"
    "                    globals()['ANTENNA_MASK_CACHE'] = {}\\n"
    "                _cache = globals()['ANTENNA_MASK_CACHE']\\n"
    "                _name = viewpoint_cam.image_name\\n"
    "                if _name not in _cache:\\n"
    "                    import numpy as _np\\n"
    "                    from PIL import Image as _PILImage\\n"
    "                    _mpath = os.path.join(antenna_mask_dir, _name + '.png')\\n"
    "                    if os.path.isfile(_mpath):\\n"
    "                        _m = _np.array(_PILImage.open(_mpath).convert('L'), dtype=_np.float32) / 255.0\\n"
    "                        _cache[_name] = torch.from_numpy(_m).unsqueeze(0).cuda()\\n"
    "                    else:\\n"
    "                        _cache[_name] = None\\n"
    "                _amask = _cache[_name]\\n"
    "                if (\\n"
    "                    _amask is not None\\n"
    "                    and _amask.shape[-2:] == image.shape[-2:]\\n"
    "                    and float(_amask.sum()) > 0\\n"
    "                ):\\n"
    "                    _denom = float(_amask.sum()) * image.shape[0] + 1e-6\\n"
    "                    Ll1_antenna = torch.abs((image - gt_image) * _amask).sum() / _denom\\n"
    "                    loss = loss + antenna_loss_weight * Ll1_antenna\\n"
    "\\n"
    "        loss.backward()\\n"
)
fallback_patch = (
    "        refine_image_loss_weight = float(os.environ.get('REFINE_IMAGE_LOSS_WEIGHT', '0') or 0)\\n"
    "        if refine_image_loss_weight > 0:\\n"
    "            if 'REFINE_IMAGE_NAME_SET' not in globals():\\n"
    "                _refine_csv = os.environ.get('REFINE_IMAGE_LIST', '')\\n"
    "                _refine_names = set()\\n"
    "                if _refine_csv and os.path.isfile(_refine_csv):\\n"
    "                    import csv as _csv\\n"
    "                    with open(_refine_csv, newline='', encoding='utf-8') as _f:\\n"
    "                        for _row in _csv.DictReader(_f):\\n"
    "                            if int(float(_row.get('refine_flag', '0') or 0)) == 1:\\n"
    "                                _refine_names.add(_row['image'])\\n"
    "                globals()['REFINE_IMAGE_NAME_SET'] = _refine_names\\n"
    "            _refine_names = globals()['REFINE_IMAGE_NAME_SET']\\n"
    "            if viewpoint_cam.image_name in _refine_names:\\n"
    "                _refine_ll1 = torch.abs(image - gt_image).mean()\\n"
    "                loss = loss + refine_image_loss_weight * _refine_ll1\\n"
    "\\n"
    "        antenna_loss_weight = float(os.environ.get('ANTENNA_LOSS_WEIGHT', '0') or 0)\\n"
    "        if antenna_loss_weight > 0:\\n"
    "            antenna_mask_dir = os.environ.get('ANTENNA_MASK_DIR', '')\\n"
    "            if antenna_mask_dir:\\n"
    "                if 'ANTENNA_MASK_CACHE' not in globals():\\n"
    "                    globals()['ANTENNA_MASK_CACHE'] = {}\\n"
    "                _cache = globals()['ANTENNA_MASK_CACHE']\\n"
    "                _name = viewpoint_cam.image_name\\n"
    "                if _name not in _cache:\\n"
    "                    import numpy as _np\\n"
    "                    from PIL import Image as _PILImage\\n"
    "                    _mpath = os.path.join(antenna_mask_dir, _name + '.png')\\n"
    "                    if os.path.isfile(_mpath):\\n"
    "                        _m = _np.array(_PILImage.open(_mpath).convert('L'), dtype=_np.float32) / 255.0\\n"
    "                        _cache[_name] = torch.from_numpy(_m).unsqueeze(0).cuda()\\n"
    "                    else:\\n"
    "                        _cache[_name] = None\\n"
    "                _amask = _cache[_name]\\n"
    "                if (\\n"
    "                    _amask is not None\\n"
    "                    and _amask.shape[-2:] == image.shape[-2:]\\n"
    "                    and float(_amask.sum()) > 0\\n"
    "                ):\\n"
    "                    _denom = float(_amask.sum()) * image.shape[0] + 1e-6\\n"
    "                    Ll1_antenna = torch.abs((image - gt_image) * _amask).sum() / _denom\\n"
    "                    loss = loss + antenna_loss_weight * Ll1_antenna\\n"
    "\\n"
    "        loss.backward()\\n"
)

if 'REFINE_IMAGE_LOSS_WEIGHT' not in text:
    if anchor in text:
        text = text.replace(anchor, patch, 1)
    elif fallback_anchor in text:
        text = text.replace(fallback_anchor, fallback_patch, 1)
    else:
        raise SystemExit('Khong tim thay anchor trong train.py -- GS_REPO co the da doi source, dung lai truoc khi train.')
    train_py.write_text(text, encoding='utf-8')
    print('Da patch train.py them refine-image + antenna extra-loss:', train_py)
else:
    print('train.py da duoc patch tu truoc, bo qua.')
"""

STAGE1_CONFIG_CELL = """# =====================================================
# CELL X - Stage A config: full-scene 30k + safe densify
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

STAGE1_SOURCE_DIR_OVERRIDE = ""
STAGE2_SOURCE_DIR_OVERRIDE = ""
REFINE_THRESHOLD_100 = "76"
MAX_REFINE_ROUNDS = "4"
REFINE_ROUND_STEP_ITERS = "5000"
REFINE_IMAGE_LOSS_WEIGHT = "1.0"
ANTENNA_LOSS_WEIGHT = "1.0"
NOISE_MASK_DIR_OVERRIDE = ""

print('=== Stage A: prepared + exposure_off + safe-densify (30k) ===')
print('Proxy no-GT se duoc dung de chon anh refine; GT chi cham cuoi.')
print('Mac dinh stage B resume tren cung source voi A; co the doi sang folder B qua STAGE2_SOURCE_DIR_OVERRIDE.')
print('Neu dung SOURCE_DIR_OVERRIDE, folder do phai co images/ va sparse/0/.')
"""

BUILD_TRAIN_MASKS_CELL = """# =====================================================
# CELL X - Sinh train-mask vung antenna/tru
# =====================================================
import subprocess

scene_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE
mask_source_dir = Path(STAGE2_SOURCE_DIR_OVERRIDE or STAGE1_SOURCE_DIR_OVERRIDE or (scene_root / 'colmap' / 'dense'))
sparse_dir = mask_source_dir / 'sparse' / '0'
bbox_json = scene_root / 'tower_bbox3d.json'
masks_dir = scene_root / 'masks' / 'tower_train'

assert sparse_dir.exists(), f'Khong thay {sparse_dir} -- can co prepared source hop le'
assert bbox_json.exists(), f'Khong thay {bbox_json}'

subprocess.run(
    [
        'python3',
        str(PROJECT_DIR / 'pipeline' / 'scripts' / 'build_tower_train_masks.py'),
        '--sparse_dir', str(sparse_dir),
        '--tower_bbox3d_json', str(bbox_json),
        '--out_dir', str(masks_dir),
        '--dilate_px', '12',
    ],
    check=True,
)
print('Mask source dir =', mask_source_dir)
print('ANTENNA_MASK_DIR =', masks_dir)
"""

BUILD_TEST_MASKS_CELL = """# =====================================================
# CELL X - Sinh test-mask cho proxy no-GT
# =====================================================
import subprocess

scene_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE
bbox_json = scene_root / 'tower_bbox3d.json'
test_mask_dir = scene_root / 'masks' / 'tower_test'

assert bbox_json.exists(), f'Khong thay {bbox_json}'
subprocess.run(
    [
        'python3',
        str(PROJECT_DIR / 'trick' / 'scripts' / 'bootstrap_tower_masks.py'),
        '--scene', SCENE,
        '--dataset_root', str(DATASET_ROOT),
        '--tower_bbox3d_json', str(bbox_json),
        '--out_dir', str(test_mask_dir),
        '--dilate_px', '12',
    ],
    check=True,
)
print('Tower test masks =', test_mask_dir)
"""

STAGE2_LOOP_CELL = """# =====================================================
# CELL X - Stage B loop: proxy no-GT -> refine den khi khong con anh < threshold
# =====================================================
import csv
import json
import os
import re
import subprocess

scene_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE
model_dir = scene_root / 'gs_model'
masks_dir = scene_root / 'masks' / 'tower_train'
test_mask_dir = scene_root / 'masks' / 'tower_test'
noise_mask_dir = Path(NOISE_MASK_DIR_OVERRIDE) if NOISE_MASK_DIR_OVERRIDE else None

assert masks_dir.exists(), f'Khong thay mask train: {masks_dir}'
assert test_mask_dir.exists(), f'Khong thay mask test: {test_mask_dir}'


def find_latest_checkpoint(model_dir: Path) -> tuple[Path, int]:
    ckpts = sorted(model_dir.glob('chkpnt*.pth'))
    if ckpts:
        latest = ckpts[-1]
        m = re.search(r'chkpnt(\\d+)\\.pth$', latest.name)
        if m:
            return latest, int(m.group(1))
    iters = sorted(
        int(p.name.split('_')[-1])
        for p in (model_dir / 'point_cloud').glob('iteration_*')
        if p.is_dir()
    )
    if not iters:
        raise SystemExit(f'Khong tim thay checkpoint/iteration trong {model_dir}')
    it = iters[-1]
    return model_dir / f'chkpnt{it}.pth', it


def count_flagged(csv_path: Path) -> int:
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return sum(int(float(r['refine_flag'])) for r in rows)


current_renders_dir = scene_root / 'renders_b2_pilot'
assert current_renders_dir.exists(), f'Khong thay render stage A: {current_renders_dir}'

final_proxy_csv = None
final_renders_dir = current_renders_dir
refine_round_history = []

for round_idx in range(1, int(MAX_REFINE_ROUNDS) + 1):
    proxy_csv = scene_root / f'proxy_scores_round_{round_idx:02d}.csv'
    proxy_json = scene_root / f'proxy_scores_round_{round_idx:02d}.json'
    refine_train_csv = scene_root / f'refine_train_views_round_{round_idx:02d}.csv'
    cmd = [
        'python3',
        str(PROJECT_DIR / 'pipeline' / 'scripts' / 'score_refine_candidates_no_gt.py'),
        '--renders_dir', str(current_renders_dir),
        '--out_csv', str(proxy_csv),
        '--summary_json', str(proxy_json),
        '--threshold_100', str(REFINE_THRESHOLD_100),
        '--tower_mask_dir', str(test_mask_dir),
        '--skyline_top_frac', '0.3',
    ]
    if noise_mask_dir is not None:
        cmd.extend(['--noise_mask_dir', str(noise_mask_dir)])
    subprocess.run(cmd, check=True)

    flagged = count_flagged(proxy_csv)
    final_proxy_csv = proxy_csv
    refine_round_history.append({'round': round_idx, 'proxy_csv': str(proxy_csv), 'flagged': flagged})
    print(f'Round {round_idx}: flagged images =', flagged)
    if flagged == 0:
        print('Khong con anh nao duoi nguong proxy -> dung vong refine.')
        break

    map_source_dir = Path(STAGE2_SOURCE_DIR_OVERRIDE or STAGE1_SOURCE_DIR_OVERRIDE or (scene_root / 'colmap' / 'dense'))
    subprocess.run(
        [
            'python3',
            str(PROJECT_DIR / 'pipeline' / 'scripts' / 'map_refine_targets_to_train_views.py'),
            '--proxy_csv', str(proxy_csv),
            '--test_poses_csv', str(DATASET_ROOT / SCENE / 'test' / 'test_poses.csv'),
            '--sparse_dir', str(map_source_dir / 'sparse' / '0'),
            '--out_csv', str(refine_train_csv),
            '--top_k', '4',
        ],
        check=True,
    )

    start_ckpt, current_iter = find_latest_checkpoint(model_dir)
    target_iter = current_iter + int(REFINE_ROUND_STEP_ITERS)
    render_out_dir = scene_root / f'renders_b2_refine_round_{round_idx:02d}'

    env = os.environ.copy()
    env['DATASET_ROOT'] = str(DATASET_ROOT)
    env['COLMAP_BIN'] = str(COLMAP_BIN)
    env['GS_REPO'] = str(GS_REPO)
    env['RUN_DENSE'] = '0'
    env['RUN_TRAIN'] = '1'
    env['RUN_RENDER'] = '1'
    env['RUN_EVAL'] = '0'
    env['SOURCE_MODE'] = 'prepared'
    env['ITERATION'] = '-1'
    env['LOW_VRAM_PROFILE'] = LOW_VRAM_PROFILE_OVERRIDE
    env['RESOLUTION'] = RESOLUTION_OVERRIDE
    env['EXPOSURE_COMP'] = EXPOSURE_COMP_OVERRIDE
    env['ITERATIONS'] = str(target_iter)
    env['START_CHECKPOINT'] = str(start_ckpt)
    env['CHECKPOINT_ITERATIONS_OVERRIDE'] = str(target_iter)
    env['SAVE_ITERATIONS_OVERRIDE'] = str(target_iter)
    env['REFINE_IMAGE_LIST'] = str(refine_train_csv)
    env['REFINE_IMAGE_LOSS_WEIGHT'] = REFINE_IMAGE_LOSS_WEIGHT
    env['ANTENNA_MASK_DIR'] = str(masks_dir)
    env['ANTENNA_LOSS_WEIGHT'] = ANTENNA_LOSS_WEIGHT
    env['RENDER_OUT_DIR'] = str(render_out_dir)
    env['RUN_STEREO'] = '0'
    if STAGE2_SOURCE_DIR_OVERRIDE:
        env['SOURCE_DIR_OVERRIDE'] = STAGE2_SOURCE_DIR_OVERRIDE

    run_and_stream(
        ['bash', str(PROJECT_DIR / 'pipeline' / 'scripts' / '05_run_b2_pilot.sh'), SCENE],
        env=env,
        log_files=[
            scene_root / 'train.log',
            scene_root / 'logs' / '05_render_b2_pilot.log',
            scene_root / 'logs' / '05_b2_pilot_summary.txt',
        ],
        poll_seconds=15,
        label=f'b2-refine-round-{round_idx:02d}',
    )
    current_renders_dir = render_out_dir
    final_renders_dir = render_out_dir

FINAL_RENDERS_DIR = final_renders_dir
FINAL_PROXY_CSV = final_proxy_csv
REFINE_ROUND_HISTORY = refine_round_history
print('FINAL_RENDERS_DIR =', FINAL_RENDERS_DIR)
print('FINAL_PROXY_CSV =', FINAL_PROXY_CSV)
print('REFINE_ROUND_HISTORY =', json.dumps(REFINE_ROUND_HISTORY, ensure_ascii=False, indent=2))
"""

FINAL_GT_EVAL_CELL = """# =====================================================
# CELL X - Cham diem THAT cua ket qua cuoi cung bang GT (psnr_max=50.0)
# =====================================================
import subprocess

scene_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE
assert FINAL_RENDERS_DIR.exists(), f'Khong thay FINAL_RENDERS_DIR: {FINAL_RENDERS_DIR}'

_, final_iter = find_latest_checkpoint(scene_root / 'gs_model')
FINAL_EVAL_CSV = scene_root / f'eval_metrics_b2_final_iter_{final_iter}.csv'
bbox_json = scene_root / 'tower_bbox3d.json'

cmd = [
    'python3',
    str(PROJECT_DIR / 'pipeline' / 'scripts' / 'eval_round1_metrics.py'),
    '--scene', SCENE,
    '--dataset_root', str(DATASET_ROOT),
    '--renders_dir', str(FINAL_RENDERS_DIR),
    '--out_csv', str(FINAL_EVAL_CSV),
    '--psnr_max', '50.0',
    '--skyline_top_frac', '0.3',
]
if bbox_json.exists():
    cmd.extend(['--tower_bbox3d_json', str(bbox_json)])
subprocess.run(cmd, check=True)
print('FINAL_EVAL_CSV =', FINAL_EVAL_CSV)
"""

REPORT_CELL = """# =====================================================
# CELL X - Bao cao cuoi: A vs B
# =====================================================
import csv
from statistics import mean

BASELINE_FULL = 0.6731
BASELINE_TOWER = 0.7064
BASELINE_SKYLINE = 0.6384

scene_root = PROJECT_DIR / 'pipeline' / 'work' / SCENE


def read_scores(csv_path):
    if not csv_path or not Path(csv_path).exists():
        return None
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    fields = ('psnr', 'ssim', 'lpips', 'score')
    return {k: mean(float(r[k]) for r in rows) for k in fields}, len(rows)


def show(label, csv_path, baseline):
    result = read_scores(csv_path)
    if result is None:
        print(f'{label:<28} : KHONG co du lieu')
        return
    means, n = result
    delta = means['score'] - baseline
    print(f'{label:<28} : n={n:<3} psnr={means["psnr"]:.4f} ssim={means["ssim"]:.4f} '
          f'lpips={means["lpips"]:.4f} score={means["score"]:.4f} (delta={delta:+.4f})')


_, final_iter = find_latest_checkpoint(scene_root / 'gs_model')
final_eval_csv = scene_root / f'eval_metrics_b2_final_iter_{final_iter}.csv'

print('=== Proxy refine history (KHONG dung GT) ===')
for item in REFINE_ROUND_HISTORY:
    print(item)
print()
print('Final proxy csv:', FINAL_PROXY_CSV)
print()

print('-- full-image GT THAT --')
print(f'{"baseline":<28} : score={BASELINE_FULL:.4f}')
show('stage1 (A, 30k)', scene_root / 'eval_metrics_b2_pilot.csv', BASELINE_FULL)
show('final (B)', final_eval_csv, BASELINE_FULL)

print()
print('-- tower-crop GT THAT --')
print(f'{"baseline":<28} : score={BASELINE_TOWER:.4f}')
show('stage1 (A, 30k)', scene_root / 'eval_metrics_b2_pilot_tower_crop.csv', BASELINE_TOWER)
show('final (B)', final_eval_csv.with_name(final_eval_csv.stem + '_tower_crop.csv'), BASELINE_TOWER)

print()
print('-- skyline-crop GT THAT --')
print(f'{"baseline":<28} : score={BASELINE_SKYLINE:.4f}')
show('stage1 (A, 30k)', scene_root / 'eval_metrics_b2_pilot_skyline_crop.csv', BASELINE_SKYLINE)
show('final (B)', final_eval_csv.with_name(final_eval_csv.stem + '_skyline_crop.csv'), BASELINE_SKYLINE)
"""


def to_source_block(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Khong tim thay doan can thay: {old[:80]!r}")
    return text.replace(old, new, 1)


def make_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": to_source_block(text)}


def build_variant(nb: dict) -> dict:
    base = copy.deepcopy(nb["cells"])

    cell0 = copy.deepcopy(base[0])
    cell0["source"] = to_source_block(
        "".join(cell0["source"]).replace(
            "# BTS Digital Twin - B2 Pilot (`hcm0031`)",
            "# BTS Digital Twin - Refine Loop No-GT + Final GT (`hcm0031`)",
        )
    )

    cell1 = copy.deepcopy(base[1])
    src1 = "".join(cell1["source"])
    src1 = replace_once(src1, "GITHUB_TOKEN = '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'\n", "GITHUB_TOKEN = ''\n")
    src1 = replace_once(
        src1,
        "BUILD_COLMAP_CUDA_IF_NEEDED = True\n",
        "BUILD_COLMAP_CUDA_IF_NEEDED = False  # no-depth refine notebook: bo qua CUDA build\n",
    )
    cell1["source"] = to_source_block(src1)

    cell6 = copy.deepcopy(base[6])
    src6 = "".join(cell6["source"])
    old6 = (
        "verify_proc = subprocess.run(\n"
        "    ['bash', '-lc', f'\"{colmap_bin_for_b2}\" -h | head -n 20'],\n"
        "    text=True,\n"
        "    capture_output=True,\n"
        "    check=False,\n"
        ")\n"
        "print(verify_proc.stdout)\n"
        "print(verify_proc.stderr)\n"
        "verify_text = (verify_proc.stdout or '') + '\\n' + (verify_proc.stderr or '')\n"
        "if 'without CUDA' in verify_text:\n"
        "    raise SystemExit('COLMAP_BIN moi van without CUDA. Khong duoc chay B2.')\n"
        "\n"
        "print('COLMAP CUDA verify PASS')\n"
    )
    new6 = (
        "if BUILD_COLMAP_CUDA_IF_NEEDED:\n"
        "    verify_proc = subprocess.run(\n"
        "        ['bash', '-lc', f'\"{colmap_bin_for_b2}\" -h | head -n 20'],\n"
        "        text=True,\n"
        "        capture_output=True,\n"
        "        check=False,\n"
        "    )\n"
        "    print(verify_proc.stdout)\n"
        "    print(verify_proc.stderr)\n"
        "    verify_text = (verify_proc.stdout or '') + '\\n' + (verify_proc.stderr or '')\n"
        "    if 'without CUDA' in verify_text:\n"
        "        raise SystemExit('COLMAP_BIN moi van without CUDA. Khong duoc chay B2.')\n"
        "    print('COLMAP CUDA verify PASS')\n"
        "else:\n"
        "    print('BUILD_COLMAP_CUDA_IF_NEEDED=False -> bo qua verify CUDA (RUN_STEREO=0).')\n"
    )
    cell6["source"] = to_source_block(replace_once(src6, old6, new6))

    cell12 = copy.deepcopy(base[12])
    cell12["source"] = to_source_block(
        replace_once(
            "".join(cell12["source"]),
            "env['RUN_EVAL'] = RUN_EVAL\n",
            "env['RUN_EVAL'] = RUN_EVAL\nenv['RUN_STEREO'] = '0'  # khong dung --depths trong notebook nay\n",
        )
    )

    config_cell = make_cell(
        "# =========================\n"
        "# CELL X - Runtime config + checks\n"
        "# =========================\n"
        "from pathlib import Path\n"
        "\n"
        "PROJECT_DIR = Path(WORKDIR) / 'project'\n"
        "DATASET_ROOT = Path(dataset_root)\n"
        "COLMAP_BIN = Path(colmap_bin_for_b2)\n"
        "GS_REPO = Path(WORKDIR) / 'gaussian-splatting'\n"
        "GS_REPO_COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'\n"
        "\n"
        "checks = [\n"
        "    PROJECT_DIR,\n"
        "    DATASET_ROOT,\n"
        "    PROJECT_DIR / 'pipeline' / 'work' / SCENE / 'colmap' / 'dense' / 'images',\n"
        "    PROJECT_DIR / 'pipeline' / 'work' / SCENE / 'colmap' / 'dense' / 'sparse' / '0',\n"
        "]\n"
        "for p in checks:\n"
        "    print(p, 'OK' if p.exists() else 'MISS')\n"
        "assert DATASET_ROOT.exists(), DATASET_ROOT\n"
        "assert COLMAP_BIN.exists(), COLMAP_BIN\n"
        "print('Notebook nay khong yeu cau fused.ply hay COLMAP CUDA build vi RUN_STEREO=0.')\n"
    )

    gs_clone_cell = make_cell(
        "# =====================================================\n"
        "# CELL X - Clone pinned Gaussian Splatting repo sach se\n"
        "# =====================================================\n"
        "import subprocess\n"
        "\n"
        "GS_REPO_URL = 'https://github.com/graphdeco-inria/gaussian-splatting.git'\n"
        "subprocess.run(['bash', '-lc', f'rm -rf \"{GS_REPO}\"'], check=True)\n"
        "subprocess.run(['git', 'clone', GS_REPO_URL, str(GS_REPO)], check=True)\n"
        "subprocess.run(['git', '-C', str(GS_REPO), 'checkout', GS_REPO_COMMIT], check=True)\n"
        "print('Cloned + pinned:', GS_REPO, GS_REPO_COMMIT)\n"
    )

    cell18 = copy.deepcopy(base[18])
    src18 = "".join(cell18["source"]).replace(
        "python3 -m pip install plyfile tqdm lpips opencv-python imageio imageio-ffmpeg",
        "python3 -m pip install numpy pillow scikit-image plyfile tqdm lpips opencv-python imageio imageio-ffmpeg",
    )
    cell18["source"] = to_source_block(src18)

    stage1_train_cell = copy.deepcopy(base[20])
    src20 = "".join(stage1_train_cell["source"]).replace("assert COLMAP_BIN.exists(), COLMAP_BIN\n", "")
    src20 = replace_once(
        src20,
        "env['SOURCE_MODE'] = 'prepared'\nenv['ITERATION'] = '-1'\n",
        (
            "env['SOURCE_MODE'] = 'prepared'\n"
            "env['ITERATION'] = '-1'\n"
            "env['LOW_VRAM_PROFILE'] = LOW_VRAM_PROFILE_OVERRIDE\n"
            "env['RESOLUTION'] = RESOLUTION_OVERRIDE\n"
            "env['EXPOSURE_COMP'] = EXPOSURE_COMP_OVERRIDE\n"
            "env['RUN_STEREO'] = '0'\n"
            "env['DENSIFY_UNTIL_ITER'] = DENSIFY_UNTIL_ITER_OVERRIDE\n"
            "env['DENSIFY_FROM_ITER'] = DENSIFY_FROM_ITER_OVERRIDE\n"
            "env['DENSIFICATION_INTERVAL'] = DENSIFICATION_INTERVAL_OVERRIDE\n"
            "env['PERCENT_DENSE'] = PERCENT_DENSE_OVERRIDE\n"
            "env['OPACITY_RESET_INTERVAL'] = OPACITY_RESET_INTERVAL_OVERRIDE\n"
            "env['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF_OVERRIDE\n"
            "if STAGE1_SOURCE_DIR_OVERRIDE:\n"
            "    env['SOURCE_DIR_OVERRIDE'] = STAGE1_SOURCE_DIR_OVERRIDE\n"
        ),
    )
    stage1_train_cell["source"] = to_source_block(src20)

    out = copy.deepcopy(nb)
    out["cells"] = [
        cell0,
        cell1,
        copy.deepcopy(base[2]),
        make_cell(BOOTSTRAP_PROJECT_CELL),
        copy.deepcopy(base[3]),
        copy.deepcopy(base[4]),
        copy.deepcopy(base[5]),
        cell6,
        copy.deepcopy(base[7]),
        copy.deepcopy(base[8]),
        copy.deepcopy(base[9]),
        copy.deepcopy(base[10]),
        copy.deepcopy(base[11]),
        cell12,
        config_cell,
        gs_clone_cell,
        cell18,
        copy.deepcopy(base[19]),
        make_cell(STAGE1_CONFIG_CELL),
        make_cell(TRAIN_PY_PATCH_CELL),
        stage1_train_cell,
        make_cell(BUILD_TRAIN_MASKS_CELL),
        make_cell(BUILD_TEST_MASKS_CELL),
        make_cell(STAGE2_LOOP_CELL),
        make_cell(FINAL_GT_EVAL_CELL),
        make_cell(REPORT_CELL),
    ]
    return out


def main() -> None:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    variant = build_variant(nb)
    out_path = ROOT / "downloads" / "B2_antenna_2stage_30k_30k.ipynb"
    out_path.write_text(json.dumps(variant, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
