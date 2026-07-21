# P0 Round1 Public

`P0` la buoc xac minh va khoi phuc du lieu benchmark `round1 public_set` truoc khi bat dau `M0`.

## Phat hien hien tai

Audit local tren `Dataset/VAI_NVS_DATA/phase1/public_set` cho thay:

- `HCM0181`, `HCM0193`, `HCM0204`, `hcm0034`: tat ca anh `train/test` deu co cung kich thuoc `262144` bytes.
- `hcm0031`: `199/200` anh train va `50/50` anh test deu co kich thuoc `524288` bytes.

Day la dau hieu rat manh cho thay du lieu local da bi cat cung theo kich thuoc file, khong con dang tin de tinh metric.

## He qua

- Khong duoc chay `M0` tren bo `round1 public_set` local hien tai.
- Can dong bo lai du lieu sach tu nguon goc/Kaggle truoc.

## Viec can lam trong P0

1. Dong bo lai toan bo scene can benchmark trong `Dataset/VAI_NVS_DATA/phase1/public_set/`.
2. Uu tien toi thieu phai phuc hoi sach `hcm0031/train/images` va `hcm0031/test/images`.
3. Sau khi dong bo xong, chay lai audit:

```bash
python3 pipeline/scripts/audit_round1_public_images.py \
  --dataset_root "Dataset/VAI_NVS_DATA/phase1/public_set" \
  --out_csv "pipeline/work/p0_round1_public_audit.csv"
```

4. Chi khi file size khong con bi dong loat cat cung moi duoc sang `M0`.

## Tieu chi dat P0

- Kich thuoc anh trong moi scene khong con dong loat mot gia tri co dinh.
- `hcm0031/test/images` khong con tinh trang `50/50` file cung mot size.
- Co the tinh lai metric va ket qua phai khop hop ly voi `eval_metrics.csv` da luu truoc do.
