"""Cells to paste into the existing Colab/Kaggle notebook after dense B2 passed.

Copy each CELL block into a separate notebook cell, in order.
Assumes these already exist in the live session:
  - /content/project
  - /content/project/.local/colmap-cuda/bin/colmap
  - /content/project/pipeline/work/hcm0031/colmap/dense/fused.ply
"""

# =========================
# CELL 1 - Config + checks
# =========================
from pathlib import Path

WORKDIR = "/content"
SCENE = "hcm0031"
PROJECT_DIR = Path(WORKDIR) / "project"
DATASET_ROOT = Path("/content/_dataset_round1_raw/VAI_NVS_DATA/phase1/public_set")
COLMAP_BIN = PROJECT_DIR / ".local" / "colmap-cuda" / "bin" / "colmap"
GS_REPO = Path("/content/gaussian-splatting")
GS_REPO_BRANCH = "main"

checks = [
    PROJECT_DIR,
    DATASET_ROOT,
    COLMAP_BIN,
    PROJECT_DIR / "pipeline" / "work" / SCENE / "colmap" / "dense" / "fused.ply",
]
for p in checks:
    print(p, "OK" if p.exists() else "MISS")


# ======================================
# CELL 2 - Clone GS repo only if missing
# ======================================
import subprocess

GS_REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting.git"

if GS_REPO.exists():
    print("GS_REPO da ton tai:", GS_REPO)
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", GS_REPO_BRANCH, GS_REPO_URL, str(GS_REPO)],
        check=True,
    )
    print("Cloned:", GS_REPO)


# ======================================================
# CELL 3 - Install minimal GS deps + CUDA extensions
# ======================================================
import subprocess
from pathlib import Path

assert GS_REPO.exists(), GS_REPO

subprocess.run(
    ["bash", "-lc", f'cd "{GS_REPO}" && git submodule update --init --recursive'],
    check=True,
)

subprocess.run(
    ["bash", "-lc", 'python -m pip install --upgrade pip setuptools wheel ninja'],
    check=True,
)

# Install only the common Python deps needed by train/render/eval.
subprocess.run(
    [
        "bash",
        "-lc",
        "python -m pip install plyfile tqdm lpips opencv-python imageio imageio-ffmpeg",
    ],
    check=True,
)

# Ensure torch is importable before building CUDA extensions.
subprocess.run(
    ["bash", "-lc", 'python -c "import torch; print(torch.__version__)"'],
    check=True,
)

build_logs_dir = Path(WORKDIR) / "gs_build_logs"
build_logs_dir.mkdir(parents=True, exist_ok=True)

build_steps = [
    (
        "diff_gaussian_rasterization",
        GS_REPO / "submodules" / "diff-gaussian-rasterization",
        build_logs_dir / "diff_gaussian_rasterization.log",
    ),
    (
        "simple_knn",
        GS_REPO / "submodules" / "simple-knn",
        build_logs_dir / "simple_knn.log",
    ),
]

for name, src_dir, log_path in build_steps:
    assert src_dir.exists(), src_dir
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            ["bash", "-lc", f'cd "{src_dir}" && MAX_JOBS=2 python -m pip install --no-build-isolation .'],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        print(f"Build fail: {name}")
        print(f"Xem log: {log_path}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        print(text[-12000:])
        raise SystemExit(f"{name} install fail, exit code={proc.returncode}")
    print(f"Installed: {name}")

print("Gaussian Splatting minimal deps + extensions installed")


# ==================================================
# CELL 4 - Quick import test for GS + eval packages
# ==================================================
import sys

sys.path.insert(0, str(GS_REPO))

import cv2
import diff_gaussian_rasterization
import lpips
import simple_knn

print("Imports OK")
print("GS_REPO =", GS_REPO)


# ================================================
# CELL 5 - Run train + render + eval for B2 debug
# ================================================
import os
import subprocess
from pathlib import Path

assert PROJECT_DIR.exists(), PROJECT_DIR
assert DATASET_ROOT.exists(), DATASET_ROOT
assert COLMAP_BIN.exists(), COLMAP_BIN
assert GS_REPO.exists(), GS_REPO

env = os.environ.copy()
env["DATASET_ROOT"] = str(DATASET_ROOT)
env["COLMAP_BIN"] = str(COLMAP_BIN)
env["GS_REPO"] = str(GS_REPO)

# Dense da xong, khong chay lai
env["RUN_DENSE"] = "0"

# Chay tiep phan con lai
env["RUN_TRAIN"] = "1"
env["RUN_RENDER"] = "1"
env["RUN_EVAL"] = "1"
env["SOURCE_MODE"] = "prepared"

# Co the doi iteration neu can
env["ITERATION"] = "-1"

cmd = ["bash", str(PROJECT_DIR / "pipeline" / "scripts" / "05_run_b2_pilot.sh"), SCENE]
proc = subprocess.run(cmd, env=env, text=True, capture_output=True)

print("RETURN CODE =", proc.returncode)
print("\n=== STDOUT ===\n")
print(proc.stdout[-8000:])
print("\n=== STDERR ===\n")
print(proc.stderr[-8000:])

if proc.returncode != 0:
    scene_root = PROJECT_DIR / "pipeline" / "work" / SCENE
    for p in [
        scene_root / "train.log",
        scene_root / "logs" / "05_render_b2_pilot.log",
        scene_root / "logs" / "05_eval_b2_pilot.log",
        scene_root / "logs" / "05_b2_pilot_summary.txt",
    ]:
        print(f"\n===== {p} =====")
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace")
            print(txt[-12000:])
        else:
            print("missing")
    raise SystemExit(f"B2 train/render/eval fail, exit code={proc.returncode}")

print("B2 train/render/eval done")


# ==========================================
# CELL 6 - Read quick logs / output presence
# ==========================================
from pathlib import Path

scene_root = Path("/content/project/pipeline/work/hcm0031")
for p in [
    scene_root / "eval_metrics_b2_pilot.csv",
    scene_root / "eval_metrics_b2_pilot_tower_crop.csv",
    scene_root / "eval_metrics_b2_pilot_skyline_crop.csv",
    scene_root / "logs" / "05_render_b2_pilot.log",
    scene_root / "logs" / "05_eval_b2_pilot.log",
]:
    print("\n====", p, "====")
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="replace")
        print(text[-4000:])
    else:
        print("missing")


# ==================================================
# CELL 7 - Compare full-image / tower / skyline M0
# ==================================================
import pandas as pd
from pathlib import Path

SCENE = "hcm0031"
work_dir = Path("/content/project/pipeline/work") / SCENE

comparisons = [
    ("full_image", work_dir / "eval_metrics.csv", work_dir / "eval_metrics_b2_pilot.csv"),
    ("tower_crop", work_dir / "eval_metrics_m0_tower_crop.csv", work_dir / "eval_metrics_b2_pilot_tower_crop.csv"),
    ("skyline_crop", work_dir / "eval_metrics_m0_skyline_crop.csv", work_dir / "eval_metrics_b2_pilot_skyline_crop.csv"),
]

metric_candidates = ["psnr", "ssim", "lpips", "score"]


def pick_row(df, label):
    if len(df) == 1:
        return df.iloc[0]
    for key in ["image_name", "filename", "name"]:
        if key in df.columns:
            raise SystemExit(f"{label} co nhieu dong theo tung anh; can file tong hop 1 dong.")
    return df.iloc[0]


rows = []

for region, baseline_path, b2_path in comparisons:
    if not baseline_path.exists() or not b2_path.exists():
        rows.append(
            {
                "region": region,
                "status": "missing",
                "baseline_file": baseline_path.name,
                "b2_file": b2_path.name,
            }
        )
        continue

    baseline_df = pd.read_csv(baseline_path)
    b2_df = pd.read_csv(b2_path)

    base = pick_row(baseline_df, f"{region} baseline")
    b2 = pick_row(b2_df, f"{region} b2")

    row = {
        "region": region,
        "status": "ok",
        "baseline_file": baseline_path.name,
        "b2_file": b2_path.name,
    }

    for metric in metric_candidates:
        if metric in baseline_df.columns and metric in b2_df.columns:
            base_val = float(base[metric])
            b2_val = float(b2[metric])
            delta = b2_val - base_val

            if metric == "lpips":
                verdict = "better" if delta < 0 else "worse" if delta > 0 else "same"
            else:
                verdict = "better" if delta > 0 else "worse" if delta < 0 else "same"

            row[f"{metric}_base"] = base_val
            row[f"{metric}_b2"] = b2_val
            row[f"{metric}_delta"] = delta
            row[f"{metric}_verdict"] = verdict

    rows.append(row)

summary_df = pd.DataFrame(rows)

display_cols = ["region", "status"]
for metric in metric_candidates:
    display_cols += [
        f"{metric}_base",
        f"{metric}_b2",
        f"{metric}_delta",
        f"{metric}_verdict",
    ]

display_df = summary_df.reindex(columns=display_cols)

for col in display_df.columns:
    if col.endswith(("_base", "_b2", "_delta")):
        display_df[col] = display_df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

print("So sanh baseline vs B2 cho 3 vung:")
print(display_df.to_string(index=False))


# ============================================================
# CELL 8 - Download important result files one by one, no zip
# ============================================================
import shutil
from pathlib import Path

WORKDIR = "/content"
SCENE = "hcm0031"

project_dir = Path(WORKDIR) / "project"
scene_root = project_dir / "pipeline" / "work" / SCENE
logs_dir = scene_root / "logs"
download_dir = Path(WORKDIR) / f"b2_downloads_{SCENE}"

if download_dir.exists():
    shutil.rmtree(download_dir)
download_dir.mkdir(parents=True, exist_ok=True)

copy_targets = [
    logs_dir / "04_colmap_dense_summary.txt",
    logs_dir / "05_b2_pilot_summary.txt",
    scene_root / "eval_metrics_b2_pilot.csv",
    scene_root / "eval_metrics_b2_pilot_tower_crop.csv",
    scene_root / "eval_metrics_b2_pilot_skyline_crop.csv",
    logs_dir / "04_patch_match_stereo.log",
    logs_dir / "04_stereo_fusion.log",
    project_dir / ".local" / "colmap-cuda" / "bin" / "colmap",
    scene_root / "colmap" / "dense" / "fused.ply",
]

prepared_files = []
for src in copy_targets:
    if src.exists():
        dst = download_dir / src.name
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        prepared_files.append((size, dst))
        print(f"OK   {src} ({size} bytes)")
    else:
        print(f"MISS {src}")

prepared_files.sort(key=lambda item: item[0])

print("\nThu tu download (nhe -> nang):")
for size, path in prepared_files:
    print(f"{path.name}: {size} bytes")

try:
    from google.colab import files

    for _, path in prepared_files:
        print(f"Downloading {path.name} ...")
        files.download(str(path))
except Exception as e:
    print("Khong goi duoc google.colab.files.download:", e)
    print("Hay tai thu cong trong thu muc:", download_dir)


# ==================================================================
# CELL 9 - Download in priority order: most important -> nice to have
# ==================================================================
import shutil
from pathlib import Path

WORKDIR = "/content"
SCENE = "hcm0031"

project_dir = Path(WORKDIR) / "project"
scene_root = project_dir / "pipeline" / "work" / SCENE
logs_dir = scene_root / "logs"
download_dir = Path(WORKDIR) / f"b2_priority_downloads_{SCENE}"

if download_dir.exists():
    shutil.rmtree(download_dir)
download_dir.mkdir(parents=True, exist_ok=True)

priority_groups = [
    (
        "P0_most_important",
        [
            logs_dir / "05_b2_pilot_summary.txt",
            scene_root / "eval_metrics_b2_pilot.csv",
            scene_root / "eval_metrics_b2_pilot_tower_crop.csv",
            scene_root / "eval_metrics_b2_pilot_skyline_crop.csv",
        ],
    ),
    (
        "P1_debug_logs",
        [
            scene_root / "train.log",
            logs_dir / "05_render_b2_pilot.log",
            logs_dir / "05_eval_b2_pilot.log",
            logs_dir / "04_colmap_dense_summary.txt",
            logs_dir / "04_patch_match_stereo.log",
            logs_dir / "04_stereo_fusion.log",
        ],
    ),
    (
        "P2_core_artifacts",
        [
            scene_root / "colmap" / "dense" / "fused.ply",
        ],
    ),
    (
        "P3_reuse_binary",
        [
            project_dir / ".local" / "colmap-cuda" / "bin" / "colmap",
        ],
    ),
]

prepared = []

for group_name, files_in_group in priority_groups:
    group_dir = download_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    for src in files_in_group:
        if src.exists():
            dst = group_dir / src.name
            shutil.copy2(src, dst)
            size = dst.stat().st_size
            prepared.append((group_name, size, dst))
            print(f"OK   [{group_name}] {src.name} ({size} bytes)")
        else:
            print(f"MISS [{group_name}] {src}")

print("\nThu tu tai ve theo uu tien:")
for group_name, size, path in prepared:
    print(f"{group_name}: {path.name} ({size} bytes)")

try:
    from google.colab import files

    current_group = None
    for group_name, _, path in prepared:
        if group_name != current_group:
            current_group = group_name
            print(f"\n=== Download group: {group_name} ===")
        print(f"Downloading {path.name} ...")
        files.download(str(path))
except Exception as e:
    print("Khong goi duoc google.colab.files.download:", e)
    print("Hay tai thu cong trong thu muc:", download_dir)
