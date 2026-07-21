# B2 Local CUDA Checklist

Checklist rất ngắn để chạy `B2` trên máy local có `COLMAP CUDA`.

## 1. Xác nhận môi trường

Chạy:

```bash
nvidia-smi
colmap -h
```

Điều kiện bắt buộc:

- `colmap -h` không được hiện `without CUDA`

## 2. Chuẩn bị repo và dataset

Checkout đúng nhánh:

```bash
git checkout coordination/round1-status
```

Đảm bảo có:

- `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031/train/images`
- `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031/train/sparse/0`
- `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031/test/images`

## 3. Chạy `P0` check nhanh

```bash
python3 pipeline/scripts/verify_round1_public_restore.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set"
```

## 4. Chạy `B2 dense pilot`

```bash
export DATASET_ROOT="$PWD/Dataset/VAI_NVS_DATA/phase1/public_set"
export COLMAP_BIN="$(which colmap)"
bash pipeline/scripts/04_run_colmap_dense.sh hcm0031
```

## 5. Kiểm tra artifact bắt buộc

Phải có:

- `pipeline/work/hcm0031/logs/04_colmap_dense_summary.txt`
- `pipeline/work/hcm0031/logs/04_patch_match_stereo.log`
- `pipeline/work/hcm0031/logs/04_stereo_fusion.log`
- `pipeline/work/hcm0031/colmap/dense/fused.ply`

## 6. Nếu muốn đi tiếp sang train từ source dense

```bash
export GS_REPO=/path/to/gaussian-splatting
export DATASET_ROOT="$PWD/Dataset/VAI_NVS_DATA/phase1/public_set"
export SOURCE_MODE=prepared
bash pipeline/scripts/03_train_3dgs.sh hcm0031
```

## 7. Nếu train xong, render và chấm lại

- render test poses
- chấm `full-image score`
- chỉ coi `B2` có giá trị nếu điểm tổng tăng thật

## Điều kiện pass

- `stereo_fusion` chạy xong
- có `fused.ply`
- train từ `prepared` chạy được
- `full-image score` tốt hơn baseline

## Điều kiện fail

- `colmap` báo thiếu CUDA
- `patch_match_stereo` fail
- có `fused.ply` nhưng điểm tổng không tăng
