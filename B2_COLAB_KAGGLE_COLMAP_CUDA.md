# B2 on Colab/Kaggle with COLMAP CUDA

Muc tieu: chay duoc nhanh `B2` trong notebook co GPU khi `colmap` cai san la
CPU-only.

## Ket luan ngan

- Co `nvidia-smi` va thay GPU **chua du**.
- Neu `colmap -h` hien `without CUDA` thi `patch_match_stereo` se fail.
- Muon chay `B2`, can thay `/usr/bin/colmap` bang mot binary `COLMAP` duoc
  build voi CUDA, roi set `COLMAP_BIN` tro vao binary do.

## Tai sao

Workflow `B2` cua repo goi:

- `colmap patch_match_stereo`
- `colmap stereo_fusion`

Buoc nghen la `patch_match_stereo`; nhan repo da fail som neu `COLMAP` khong co
CUDA.

## Cach kha thi nhat trong notebook

### Cach 1: build `COLMAP` tu source ngay trong notebook

Repo da co script:

`pipeline/scripts/build_colmap_cuda_notebook.sh`

Vi du cho Tesla T4 (compute capability `75`):

```bash
export CMAKE_CUDA_ARCHITECTURES=75
bash pipeline/scripts/build_colmap_cuda_notebook.sh
export COLMAP_BIN="$PWD/.local/colmap-cuda/bin/colmap"
```

Verify:

```bash
"$COLMAP_BIN" -h
```

Dieu kien dat:

- output khong con chu `without CUDA`

Sau do chay `B2`:

```bash
export DATASET_ROOT="$PWD/Dataset/VAI_NVS_DATA/phase1/public_set"
export GS_REPO=/path/to/gaussian-splatting
bash pipeline/scripts/05_run_b2_pilot.sh hcm0031
```

### Cach 2: dung binary/installation co san nhung phai verify lai

Neu ban da co mot binary `colmap` build san voi CUDA o noi khac, chi can:

```bash
export COLMAP_BIN=/path/to/that/colmap
"$COLMAP_BIN" -h
```

Chi dung binary do neu output khong hien `without CUDA`.

## Khi nao Colab/Kaggle van se fail

- Co GPU nhung `colmap -h` van hien `without CUDA`
- Co `colmap` moi nhung build ra van khong link duoc CUDA
- Notebook het thoi gian / RAM / disk truoc khi build xong

## Ghi chu thuc te cho Colab/Kaggle

- Build tu source trong notebook la cach thuc dung nhat voi repo nay vi
  `B2` can CLI `colmap`, khong chi `pycolmap`.
- Compile co the ton kha lau va rat ton disk; nen cache thu muc
  `.local/colmap-cuda/` neu muon tai su dung giua cac session.
- Neu notebook dang la Ubuntu `22.04` va build loi lien quan GCC/CUDA, script
  se uu tien `gcc-10/g++-10` neu may co san.

## Cach doc nhanh log truoc khi mat thoi gian

1. `nvidia-smi`
2. `colmap -h`
3. neu thay `without CUDA` thi dung ngay, build lai `COLMAP`
4. verify binary moi bang `"$COLMAP_BIN" -h`
5. chi luc do moi chay `04_run_colmap_dense.sh` hoac `05_run_b2_pilot.sh`
