"""Chạy COLMAP (SfM) hoàn toàn bằng pycolmap — không cần cài `colmap` CLI riêng.

Các hàm/enums dùng ở đây (pycolmap.ImageReaderOptions, pycolmap.CameraMode,
pycolmap.extract_features, match_sequential/match_exhaustive, incremental_mapping,
undistort_images, pycolmap.Reconstruction) đã được đối chiếu trực tiếp với source
C++ / pybind hiện tại của COLMAP (repo colmap/colmap, nhánh main, thư mục
src/pycolmap/pipeline/*.cc và src/pycolmap/scene/*.cc) để đảm bảo đúng tên tham số,
KHÔNG suy đoán từ trí nhớ.

Phát hiện quan trọng khi soi dữ liệu thật (xem Dataset/README.md mục 4): sparse
gốc của BTC cho scene HCM0249 dùng camera model SIMPLE_RADIAL (model_id=2, 4 tham
số f,cx,cy,k — giải mã trực tiếp từ cameras.bin, không suy đoán), khớp với việc
mọi hàng test_poses.csv luôn có fx==fy (SIMPLE_RADIAL chỉ có 1 focal length dùng
chung cho cả 2 trục). Vì vậy mặc định script này cũng dùng SIMPLE_RADIAL khi tự
chạy COLMAP trên ảnh train (ảnh gốc CÓ thể có méo ống kính nhẹ), rồi undistort
sang PINHOLE sạch (không méo) trước khi đưa vào 3D Gaussian Splatting — đúng quy
trình chuẩn mà chính script convert.py của graphdeco-inria/gaussian-splatting dùng
(feature_extractor -> matcher -> mapper -> image_undistorter).
"""
from pathlib import Path

import pycolmap

from common.logging_utils import FileLog, quiet_pycolmap


def _undistort_and_fix_layout(sparse_dir: Path, images_dir: Path, dense_dir: Path) -> None:
    """undistort_images ghi thẳng vào <dense_dir>/sparse/*.bin (không có "0/"),
    trong khi graphdeco-inria/gaussian-splatting cần <source>/sparse/0/*.bin —
    dùng chung cho cả 2 đường (tự chạy COLMAP / dùng thẳng sparse có sẵn)."""
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=sparse_dir,
        image_path=images_dir,
        output_type="COLMAP",
    )
    flat_sparse = dense_dir / "sparse"
    nested_sparse = flat_sparse / "0"
    if flat_sparse.exists() and not nested_sparse.exists():
        tmp = dense_dir / "_sparse_tmp"
        flat_sparse.rename(tmp)
        tmp_nested = dense_dir / "sparse"
        tmp_nested.mkdir(parents=True)
        tmp.rename(tmp_nested / "0")


def use_provided_sparse(images_dir: Path, sparse_dir: Path, workdir: Path) -> dict:
    """Dùng THẲNG sparse đã có sẵn (do BTC cung cấp) — KHÔNG tự chạy COLMAP.

    Kể từ khi BTC cập nhật lại dataset (xem Dataset/README.md), cả 13/13 scene
    đều có sparse/0/ hợp lệ, nên bước feature extraction + matching + incremental
    mapping (vốn tốn thời gian và dễ lỗi nhất) không còn cần thiết cho hầu hết
    trường hợp — chỉ còn bước undistort (rất nhanh, vài giây tới vài chục giây)
    để chuyển sang PINHOLE sạch trước khi đưa vào 3D Gaussian Splatting.

    Trả về dict cùng format với run_colmap_scene(), thêm "used_provided_sparse": True.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "colmap.log"
    log = FileLog(log_path)
    quiet_pycolmap(log_dir=workdir / "pycolmap_internal_logs")

    rec = pycolmap.Reconstruction(sparse_dir)
    log.write(f"Dùng sparse có sẵn: {sparse_dir} "
              f"({rec.num_reg_images()} ảnh, {rec.num_points3D()} điểm) — bỏ qua bước tự chạy COLMAP.")

    dense_dir = workdir / "dense"
    log.write(f"Undistort ảnh + camera model -> PINHOLE sạch tại {dense_dir} ...")
    _undistort_and_fix_layout(sparse_dir, images_dir, dense_dir)

    log.write(f"Xong. num_reg_images={rec.num_reg_images()} num_points3D={rec.num_points3D()}")
    log.close()

    return {
        "sparse_dir": sparse_dir,
        "dense_dir": dense_dir,
        "num_reg_images": rec.num_reg_images(),
        "num_points3D": rec.num_points3D(),
        "log_path": log_path,
        "used_provided_sparse": True,
    }


def run_colmap_scene(
    images_dir: Path,
    workdir: Path,
    matching: str = "sequential",
    camera_model: str = "SIMPLE_RADIAL",
    camera_params_prior: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Chạy toàn bộ pipeline SfM cho 1 scene.

    Các bước trung gian ([1/4]...[4/4]) chỉ ghi vào file log (không in ra
    console) — xem "log_path" trong dict trả về nếu cần tra cứu chi tiết.

    Trả về dict {
        "sparse_dir": Path,      # workdir/sparse/<best_idx>  (định dạng COLMAP gốc, có thể méo)
        "dense_dir": Path,       # workdir/dense              (images/ + sparse/0/, đã undistort PINHOLE)
        "num_reg_images": int,
        "num_points3D": int,
        "log_path": Path,
    }
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "colmap.log"
    log = FileLog(log_path)
    quiet_pycolmap(log_dir=workdir / "pycolmap_internal_logs")

    database_path = workdir / "database.db"
    if overwrite and database_path.exists():
        database_path.unlink()

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model
    if camera_params_prior:
        reader_options.camera_params = camera_params_prior

    if not database_path.exists():
        log.write(f"[1/4] Feature extraction ({images_dir.name}, model={camera_model}) ...")
        pycolmap.extract_features(
            database_path=database_path,
            image_path=images_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
        )
    else:
        log.write("[1/4] Bỏ qua feature extraction (database.db đã tồn tại).")

    log.write(f"[2/4] Feature matching ({matching}) ...")
    if matching == "exhaustive":
        pycolmap.match_exhaustive(database_path)
    else:
        pycolmap.match_sequential(database_path)

    sparse_root = workdir / "sparse"
    sparse_root.mkdir(exist_ok=True)
    if any(sparse_root.iterdir()) and not overwrite:
        log.write("[3/4] Bỏ qua mapping (sparse/ đã có kết quả).")
        recs = {
            int(p.name): pycolmap.Reconstruction(p)
            for p in sorted(sparse_root.iterdir()) if p.is_dir() and (p / "cameras.bin").exists()
        }
    else:
        log.write("[3/4] Incremental mapping (SfM + bundle adjustment) ...")
        recs = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=sparse_root,
        )

    if not recs:
        log.close()
        raise RuntimeError(
            f"COLMAP không tạo được reconstruction nào cho {images_dir}. "
            f"Thử lại với matching='exhaustive' hoặc kiểm tra chất lượng/độ chồng lấn ảnh. "
            f"Chi tiết: {log_path}"
        )

    best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
    best_rec = recs[best_idx]
    if len(recs) > 1:
        sizes = {i: r.num_reg_images() for i, r in recs.items()}
        # Quan trọng — vẫn in ra console (không chỉ ghi log), vì ảnh hưởng trực
        # tiếp tới độ đầy đủ của scene.
        print(f"[CẢNH BÁO] {images_dir.parent.parent.name}: COLMAP tách ra {len(recs)} model rời rạc: {sizes}. "
              f"Dùng model {best_idx} (nhiều ảnh nhất: {best_rec.num_reg_images()}/{len(list(images_dir.iterdir()))}). "
              f"Các ảnh ở model khác sẽ KHÔNG có trong sparse cuối cùng.")
        log.write(f"[CẢNH BÁO] tách {len(recs)} model rời rạc: {sizes}")

    sparse_dir = sparse_root / str(best_idx)

    dense_dir = workdir / "dense"
    log.write(f"[4/4] Undistort ảnh + camera model -> PINHOLE sạch tại {dense_dir} ...")
    _undistort_and_fix_layout(sparse_dir, images_dir, dense_dir)

    log.write(f"Xong. num_reg_images={best_rec.num_reg_images()} num_points3D={best_rec.num_points3D()}")
    log.close()

    return {
        "sparse_dir": sparse_dir,
        "dense_dir": dense_dir,
        "num_reg_images": best_rec.num_reg_images(),
        "num_points3D": best_rec.num_points3D(),
        "log_path": log_path,
        "used_provided_sparse": False,
    }
