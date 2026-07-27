#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import zipfile
from pathlib import Path


DEFAULT_WORK_CANDIDATES = (
    Path("/content/project/pipeline/work"),
    Path("/kaggle/working/project/pipeline/work"),
    Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/pipeline/work"),
)

DEFAULT_GS_REPO_CANDIDATES = (
    Path("/content/gaussian-splatting"),
    Path("/content/project/gaussian-splatting"),
    Path("/content/project/submodules/gaussian-splatting"),
    Path("/kaggle/working/gaussian-splatting"),
    Path("/kaggle/working/project/gaussian-splatting"),
    Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/gaussian-splatting"),
)

DEFAULT_DATASET_ROOT_CANDIDATES = (
    Path("/content/project/Dataset/VAI_NVS_DATA/phase1/public_set"),
    Path("/content/_dataset_round1_raw/VAI_NVS_DATA/phase1/public_set"),
    Path("/kaggle/working/project/Dataset/VAI_NVS_DATA/phase1/public_set"),
    Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin/Dataset/VAI_NVS_DATA/phase1/public_set"),
)


def human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    unit_idx = 0
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def size_gb(size_bytes: int) -> float:
    return size_bytes / (1024 ** 3)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_zip_from_files(base_root: Path, files: list[Path], zip_path: Path) -> None:
    ensure_parent(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            zf.write(path, arcname=path.relative_to(base_root))


def build_zip_from_dir(src_dir: Path, zip_path: Path) -> None:
    ensure_parent(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(src_dir.parent))


def write_manifest(rows: list[tuple[int, Path]], out_csv: Path) -> None:
    ensure_parent(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["size_bytes", "size_human", "path"])
        for size, path in rows:
            writer.writerow([size, human_size(size), str(path)])


def find_scene_root(scene: str, work_root: str | None) -> Path:
    candidates = [Path(work_root)] if work_root else list(DEFAULT_WORK_CANDIDATES)
    for base in candidates:
        scene_root = base / scene
        if scene_root.exists():
            return scene_root
    raise FileNotFoundError(f"Khong tim thay thu muc work cua scene {scene}")


def project_root_from_scene(scene_root: Path) -> Path:
    return scene_root.parent.parent.parent


def find_gs_repo(gs_repo: str | None) -> Path:
    if gs_repo:
        repo = Path(gs_repo)
        if (repo / "train.py").exists():
            os.environ["GS_REPO"] = str(repo)
            return repo
        raise FileNotFoundError(f"GS_REPO sai: {repo}")

    env_path = os.environ.get("GS_REPO", "")
    if env_path:
        env_repo = Path(env_path)
        if (env_repo / "train.py").exists():
            return env_repo

    for path in DEFAULT_GS_REPO_CANDIDATES:
        if (path / "train.py").exists():
            os.environ["GS_REPO"] = str(path)
            return path

    raise FileNotFoundError("Khong tu dong tim thay GS_REPO.")


def find_dataset_root(scene: str, dataset_root: str | None) -> Path:
    env_path = dataset_root or os.environ.get("DATASET_ROOT", "")
    if env_path:
        env_root = Path(env_path)
        if (env_root / scene / "test" / "test_poses.csv").exists():
            os.environ["DATASET_ROOT"] = str(env_root)
            return env_root

    for path in DEFAULT_DATASET_ROOT_CANDIDATES:
        if (path / scene / "test" / "test_poses.csv").exists():
            os.environ["DATASET_ROOT"] = str(path)
            return path

    search_bases = (
        Path("/content"),
        Path("/content/_dataset_round1_raw"),
        Path("/kaggle/working"),
        Path("/home/thongluc/Khóa Luận Tốt Nghiệp/BTS Digital Twin"),
    )
    for base in search_bases:
        if not base.exists():
            continue
        for path in base.rglob("public_set"):
            if (path / scene / "test" / "test_poses.csv").exists():
                os.environ["DATASET_ROOT"] = str(path)
                return path

    raise FileNotFoundError(f"Khong tu dong tim thay dataset root cho scene {scene}")


def gather_minimal_light_files(scene_root: Path, scene: str) -> list[Path]:
    project_root = project_root_from_scene(scene_root)
    gs_model = scene_root / "gs_model"
    sparse0 = scene_root / "colmap" / "dense" / "sparse" / "0"
    candidates = [
        gs_model / "cfg_args",
        gs_model / "cameras.json",
        gs_model / "exposure.json",
        gs_model / "pipeline_train_flags.json",
        scene_root / "eval_metrics_b2_pilot.csv",
        project_root / "pipeline" / "scripts" / "03_train_3dgs.sh",
        project_root / "pipeline" / "scripts" / "05_run_b2_pilot.sh",
        project_root / "pipeline" / "scripts" / "render_round1_test_poses.py",
        project_root / "pipeline" / "scripts" / "eval_round1_metrics.py",
        project_root / "trick" / scene / "default.env",
        sparse0 / "cameras.bin",
        sparse0 / "frames.bin",
        sparse0 / "images.bin",
        sparse0 / "points3D.bin",
        sparse0 / "points3D.ply",
        sparse0 / "rigs.bin",
    ]
    return [path for path in candidates if path.exists() and path.is_file()]


def b2_required_paths(scene_root: Path, scene: str) -> list[Path]:
    curated_root = scene_root / "_curated_b2_minimal"
    dense_sparse0 = scene_root / "colmap" / "dense" / "sparse" / "0"
    return [
        scene_root / "gs_model" / "chkpnt30000.pth",
        scene_root / "gs_model" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        scene_root / "gs_model" / "cfg_args",
        scene_root / "gs_model" / "cameras.json",
        scene_root / "gs_model" / "exposure.json",
        scene_root / "eval_metrics_b2_pilot.csv",
        scene_root / "colmap" / "dense" / "fused.ply",
        dense_sparse0 / "cameras.bin",
        dense_sparse0 / "frames.bin",
        dense_sparse0 / "images.bin",
        dense_sparse0 / "points3D.bin",
        dense_sparse0 / "points3D.ply",
        dense_sparse0 / "rigs.bin",
        project_root_from_scene(scene_root) / "pipeline" / "scripts" / "03_train_3dgs.sh",
        project_root_from_scene(scene_root) / "pipeline" / "scripts" / "05_run_b2_pilot.sh",
        project_root_from_scene(scene_root) / "pipeline" / "scripts" / "render_round1_test_poses.py",
        project_root_from_scene(scene_root) / "pipeline" / "scripts" / "eval_round1_metrics.py",
        project_root_from_scene(scene_root) / "trick" / scene / "default.env",
        curated_root / f"{scene}_b2_light.zip",
        curated_root / "colmap_dense_images.zip",
        curated_root / "renders_images.zip",
    ]


def print_b2_audit(scene_root: Path, scene: str) -> None:
    required = b2_required_paths(scene_root, scene)
    total_existing = 0
    missing = []

    print("=== B2 AUDIT CHECKLIST ===")
    for path in required:
        if path.exists() and path.is_file():
            size = path.stat().st_size
            total_existing += size
            print(f"[OK]   {human_size(size):>10}  {path}")
        else:
            missing.append(path)
            print(f"[MISS] {'-':>10}  {path}")

    print("\n=== B2 AUDIT SUMMARY ===")
    print("required_count   :", len(required))
    print("missing_count    :", len(missing))
    print("existing_total   :", human_size(total_existing))
    print(f"existing_total_gb: {size_gb(total_existing):.3f} GB")
    print("b2_ready         :", "YES" if not missing else "NO")

    if missing:
        print("\n=== THIEU CHO B2 ===")
        for path in missing:
            print(path)


def package_b2_minimal(scene_root: Path, scene: str, include_renders_archive: bool) -> None:
    curated_root = scene_root / "_curated_b2_minimal"
    curated_root.mkdir(parents=True, exist_ok=True)

    project_root = project_root_from_scene(scene_root)
    light_files = gather_minimal_light_files(scene_root, scene)
    light_zip = curated_root / f"{scene}_b2_light.zip"
    build_zip_from_files(project_root, light_files, light_zip)

    heavy_files = [
        scene_root / "gs_model" / "chkpnt30000.pth",
        scene_root / "gs_model" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
    ]
    heavy_files = [path for path in heavy_files if path.exists() and path.is_file()]

    archives: list[Path] = []
    dense_images_dir = scene_root / "colmap" / "dense" / "images"
    if dense_images_dir.exists():
        dense_images_zip = curated_root / "colmap_dense_images.zip"
        build_zip_from_dir(dense_images_dir, dense_images_zip)
        archives.append(dense_images_zip)

    renders_dir = scene_root / "renders_b2_pilot"
    if not renders_dir.exists():
        renders_dir = scene_root / "renders"
    if include_renders_archive and renders_dir.exists():
        renders_zip = curated_root / "renders_images.zip"
        build_zip_from_dir(renders_dir, renders_zip)
        archives.append(renders_zip)

    light_rows = sorted(((path.stat().st_size, path) for path in light_files), key=lambda item: (item[0], str(item[1])))
    heavy_rows = sorted(((path.stat().st_size, path) for path in heavy_files), key=lambda item: (item[0], str(item[1])))
    write_manifest(light_rows, curated_root / "light_bundle_manifest.csv")
    write_manifest(heavy_rows, curated_root / "heavy_files_manifest.csv")

    print("=== B2 MINIMAL PACKAGE ===")
    print("scene_root        :", scene_root)
    print("curated_root      :", curated_root)
    print("light_zip         :", light_zip)
    print("light_zip_size    :", human_size(light_zip.stat().st_size))
    print(f"light_zip_gb      : {size_gb(light_zip.stat().st_size):.3f} GB")

    print("\n=== KEEP ===")
    print(light_zip)
    for path in archives:
        print(path)
    for path in heavy_files:
        print(path)

    print("\n=== KHONG CAN TAI ===")
    print("colmap/dense/stereo/depth_maps/*")
    print("colmap/dense/stereo/normal_maps/*")
    print("colmap/dense/fused.ply.vis")
    print("gs_model/events.out.tfevents*")
    print("train.log")
    print("gs_model/input.ply")
    print("logs/*")
    print("_download_ready/all_filtered/* neu da co bo curated moi")

    total_downloads = [light_zip] + archives + heavy_files
    total_bytes = sum(path.stat().st_size for path in total_downloads)
    print("\n=== TOTAL DOWNLOAD ===")
    print("count             :", len(total_downloads))
    print("total_size        :", human_size(total_bytes))
    print(f"total_size_gb     : {size_gb(total_bytes):.3f} GB")


def find_latest_iteration(scene_root: Path) -> int:
    point_cloud_root = scene_root / "gs_model" / "point_cloud"
    iterations = []
    for path in point_cloud_root.glob("iteration_*"):
        if path.is_dir():
            try:
                iterations.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
    if not iterations:
        raise FileNotFoundError(f"Khong tim thay iteration_* trong {point_cloud_root}")
    return max(iterations)


def reevaluate_latest_scene(scene_root: Path, scene: str, dataset_root: str | None, gs_repo: str | None, psnr_max: float) -> None:
    project_root = project_root_from_scene(scene_root)
    gs_repo_path = find_gs_repo(gs_repo)
    dataset_root_path = find_dataset_root(scene, dataset_root)
    scripts_dir = project_root / "pipeline" / "scripts"
    logs_dir = scene_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    latest_iter = find_latest_iteration(scene_root)
    model_dir = scene_root / "gs_model"
    render_out_dir = scene_root / f"renders_b2_latest_iter_{latest_iter}"
    eval_out_csv = scene_root / f"eval_metrics_b2_latest_iter_{latest_iter}.csv"
    render_log = logs_dir / f"05_render_b2_latest_iter_{latest_iter}.log"
    eval_log = logs_dir / f"05_eval_b2_latest_iter_{latest_iter}.log"

    render_cmd = [
        "python3",
        str(scripts_dir / "render_round1_test_poses.py"),
        "--scene", scene,
        "--dataset_root", str(dataset_root_path),
        "--model_dir", str(model_dir),
        "--iteration", str(latest_iter),
        "--out_dir", str(render_out_dir),
    ]
    eval_cmd = [
        "python3",
        str(scripts_dir / "eval_round1_metrics.py"),
        "--scene", scene,
        "--dataset_root", str(dataset_root_path),
        "--renders_dir", str(render_out_dir),
        "--out_csv", str(eval_out_csv),
        "--psnr_max", str(psnr_max),
    ]

    env = os.environ.copy()
    env["GS_REPO"] = str(gs_repo_path)
    env["DATASET_ROOT"] = str(dataset_root_path)

    print("=== REEVAL LATEST SCENE ===")
    print("scene             :", scene)
    print("scene_root        :", scene_root)
    print("dataset_root      :", dataset_root_path)
    print("gs_repo           :", gs_repo_path)
    print("latest_iteration  :", latest_iter)
    print("render_out_dir    :", render_out_dir)
    print("eval_out_csv      :", eval_out_csv)
    print("psnr_max          :", psnr_max)

    with open(render_log, "w", encoding="utf-8") as f:
        subprocess.run(render_cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    with open(eval_log, "w", encoding="utf-8") as f:
        subprocess.run(eval_cmd, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)

    print("render_log        :", render_log)
    print("eval_log          :", eval_log)

    with open(eval_out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("CSV eval rong, khong co metric nao.")

    psnr_vals = [float(row["psnr"]) for row in rows]
    ssim_vals = [float(row["ssim"]) for row in rows]
    lpips_vals = [float(row["lpips"]) for row in rows]
    score_vals = [float(row["score"]) for row in rows]

    print("\n=== EVAL SUMMARY ===")
    print("images            :", len(rows))
    print(f"PSNR mean         : {sum(psnr_vals) / len(psnr_vals):.4f}")
    print(f"SSIM mean         : {sum(ssim_vals) / len(ssim_vals):.4f}")
    print(f"LPIPS mean        : {sum(lpips_vals) / len(lpips_vals):.4f}")
    print(f"Score mean        : {sum(score_vals) / len(score_vals):.4f}")

    worst_rows = sorted(rows, key=lambda row: float(row["score"]))[:10]
    print("\n=== TOP 10 ANH TE NHAT ===")
    for idx, row in enumerate(worst_rows, start=1):
        print(
            f"{idx:>2}. {row['image']} | "
            f"score={float(row['score']):.4f} | "
            f"psnr={float(row['psnr']):.4f} | "
            f"ssim={float(row['ssim']):.4f} | "
            f"lpips={float(row['lpips']):.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hcm0031")
    ap.add_argument("--mode", choices=["audit", "package", "reeval_latest"], required=True)
    ap.add_argument("--work_root", default=None, help="Thu muc cha cua cac scene trong pipeline/work")
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--gs_repo", default=None)
    ap.add_argument("--psnr_max", type=float, default=50.0)
    ap.add_argument("--include_renders_archive", action="store_true")
    args = ap.parse_args()

    scene_root = find_scene_root(args.scene, args.work_root)
    if args.mode == "audit":
        print_b2_audit(scene_root, args.scene)
        return
    if args.mode == "package":
        package_b2_minimal(scene_root, args.scene, args.include_renders_archive)
        return
    reevaluate_latest_scene(scene_root, args.scene, args.dataset_root, args.gs_repo, args.psnr_max)


if __name__ == "__main__":
    main()
