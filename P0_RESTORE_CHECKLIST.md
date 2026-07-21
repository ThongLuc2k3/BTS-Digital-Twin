# P0 Restore Checklist

Muc tieu: khoi phuc bo `round1 public_set` sach truoc khi bat dau `M0`.

Phan vi uu tien:

- `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031/train/images`
- `Dataset/VAI_NVS_DATA/phase1/public_set/hcm0031/test/images`

Phan vi khuyen nghi:

- dong bo lai toan bo `Dataset/VAI_NVS_DATA/phase1/public_set/`

Ly do:

- audit hien tai cho thay tat ca scene `round1 public_set` local deu co dau hieu anh bi cat cung theo kich thuoc file.

## Checklist truoc khi copy lai

1. Xac dinh nguon sach
- Uu tien: Kaggle run/working noi da tung train-render-eval thanh cong.
- Hoac nguon dataset goc da xac minh.

2. Khong copy de chen file tung phan
- Nen xoa/doi ten thu muc cu truoc khi copy lai de tranh tron file hong va file sach.

3. Chon cach copy giu nguyen file goc
- Uu tien dung `rsync -av`, `cp -a`, zip/tar roi giai nen.
- Tranh cac cach dong bo co the cat file theo buffer co dinh.

## Checklist copy lai

1. Backup bo local cu neu can doi chieu
- Vi du doi ten:
```bash
mv Dataset/VAI_NVS_DATA/phase1/public_set Dataset/VAI_NVS_DATA/phase1/public_set_corrupted_2026-07-21
```

2. Copy bo sach vao dung vi tri
- Dich mong muon:
```text
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/train/images
Dataset/VAI_NVS_DATA/phase1/public_set/<SCENE>/test/images
```

3. Kiem tra so luong file toi thieu
- `hcm0031/train/images`: `200`
- `hcm0031/test/images`: `50`

## Verify sau khi copy lai

Chay audit tong quat:

```bash
python3 pipeline/scripts/audit_round1_public_images.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set" \
  --out_csv "pipeline/work/p0_round1_public_audit_after_restore.csv"
```

Chay verify PASS/FAIL:

```bash
python3 pipeline/scripts/verify_round1_public_restore.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set" \
  --out_csv "pipeline/work/p0_round1_public_verify.csv"
```

## Tieu chi PASS cua P0

1. `verify_round1_public_restore.py` tra ve `P0 verify: PASS`
2. Khong con scene nao co pattern:
- toan bo file cung 1 kich thuoc
- hoac >95% file cung 1 kich thuoc
3. `hcm0031/test/images` khong con `50/50` file cung mot size
4. `hcm0031/train/images` khong con `199/200` file cung mot size

## Sau khi P0 PASS

Chi khi P0 PASS moi duoc chuyen sang:

- `M0`: dung metric full-image + skyline crop + tower crop
- sau do moi den `B2 -> C/F -> A`

## Neu P0 FAIL

1. Dung lai, khong chay `M0`
2. Kiem tra lai cach copy / nguon du lieu
3. Copy lai tu nguon sach khac neu can
