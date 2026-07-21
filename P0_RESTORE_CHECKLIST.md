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

### M0 - tower crop + skyline crop (2026-07-21 -> 2026-07-22)

`hcm0031` da PASS P0. Ca 3 loai metric cua `M0` (full-image, tower-crop,
skyline-crop) da co code chay duoc va da co so lieu that tren `hcm0031`.

**Cong cu:**

- `pipeline/scripts/estimate_object_bbox3d.py`: dung sparse COLMAP
  (`train/sparse/0`) + nhieu anh moc (moi anh 1 khung pixel khoanh quanh tru +
  1 nguong do sau `depth_cutoff` de loai diem nen lan qua khe ho cua ket cau
  luoi thep) de gom tap diem 3D cua tru. Sau do dung **PCA lap lai** (khong
  loc AABB theo truc world X/Y/Z doc lap) de tim huong nghieng that cua tru va
  cat outlier theo dung huong do - quan trong vi world axes cua COLMAP KHONG
  co truc nao la "len" chuan, tru co the nghieng cheo ca 3 truc world. Output
  la **oriented bounding box (OBB)**: 8 goc + tam + 3 truc PCA, luu trong
  `pipeline/work/<scene>/tower_bbox3d.json`.
- `pipeline/scripts/eval_round1_metrics.py`: them `--tower_bbox3d_json`
  (+ `--tower_margin_frac`) va `--skyline_top_frac`.
  - Tower-crop: chieu 8 goc OBB vao tung pose rieng trong `test_poses.csv`
    (dung cong thuc pinhole/quaternion nhu `render_round1_test_poses.py`,
    KHONG phu thuoc pycolmap o buoc eval) -> khung crop khac nhau cho tung
    anh, vi tri tru tren man hinh doi rat manh giua cac goc quay nen 1 khung
    pixel co dinh se sai o phan lon anh.
  - Skyline-crop: don gian hon, lay dai tren cung `skyline_top_frac` chieu
    cao anh (khong phu thuoc pose). Day la **gia dinh cua Claude, chua doi
    chieu voi de bai/rubric that cua BTC** - can nguoi phu trach xac nhan lai
    dinh nghia "skyline" trong de bai truoc khi dung so nay de quyet dinh.
  - Ca 2 deu ghi CSV rieng: `<out_csv>_tower_crop.csv`,
    `<out_csv>_skyline_crop.csv`.

**So lieu that tren `hcm0031`** (chay xong het tren CPU local, ~1.5 phut, KHONG
can Kaggle GPU - dung venv rieng cai `torch/torchvision` ban CPU tu
`download.pytorch.org/whl/cpu` roi moi cai `lpips`, tranh loi mismatch ABI
`torchvision::nms` khi 2 goi cai lech version nhau):

| Vung | PSNR | SSIM | LPIPS | Score | Dien tich TB |
|---|---|---|---|---|---|
| Full-image | 21.694 | 0.6819 | 0.1542 | 0.6731 | 100% |
| Tower-crop (OBB, 5 anh moc) | 23.291 | 0.7287 | 0.1298 | 0.7064 | 29.3% |
| Skyline-crop (top 30%) | 20.429 | 0.6298 | 0.1829 | 0.6384 | 30% (co dinh) |

CSV chi tiet: `pipeline/work/hcm0031/eval_metrics_m0.csv`,
`pipeline/work/hcm0031/eval_metrics_m0_tower_crop.csv`,
`pipeline/work/hcm0031/eval_metrics_m0_skyline_crop.csv`.

Ca 3 metric (PSNR/SSIM/LPIPS) deu cung chieu: tower-crop tot hon full-image,
skyline-crop te hon full-image - cung cap them bang chung (khong chi PSNR/SSIM
truoc do) rang crop dang bounding-box van con lan nhieu pixel nen tot xung
quanh tru, chua phai mask pixel that.

**Boi canh (2026-07-22):** da xac nhan voi user - Round 1 (deadline 30/07,
diem da nop that 58.67320 tren `private_set1`) **khong con la muc tieu nop bai
that nua**, phan nay chi de khoi phuc/doi chieu. M0 lam de tu tinh diem kho
khan, phuc vu quyet dinh buoc tiep theo (`B2`), khong phai de chuan bi nop.

**Dien giai (quan trong, doc truoc khi dung so):**

- Tower-crop OBB da tighten tu 2 anh moc (dien tich TB 72.3%) len 5 anh moc +
  PCA lap lai (dien tich TB con **29.3%**) - validate bang cach chieu box len
  8 anh test va xem truc tiep, box om khit quanh tru + panel anten.
- Tower-crop PSNR/SSIM **cao hon** full-image, dieu nay nguoc voi cam nhan
  truc quan (o khung diem thap nhat, luoi thep + anten bi ghosting/blur ro
  ret). Ly do: day van la **bounding box hinh chu nhat**, khong phai mask
  pixel that cua tru - vung crop van chua nhieu pixel nen/mai nha xung quanh
  va xuyen qua khe ho cua ket cau luoi (tru dang luoi rat thua), nen trung
  binh van bi keo len boi phan nen de tai tao. Muon co con so "tower dang tin"
  hon nua thi can mask pixel that (segmentation) thay vi bounding box - chi
  huong phat trien tiep, chua lam.
- Skyline-crop (top 30%) PSNR/SSIM **thap hon** full-image, khop voi quan sat
  truc quan (floater/nhoe o vung nen xa trong cac khung gan/thap).

**Con thieu de goi la M0 hoan tat:**

1. Chay lai tren Kaggle (co GPU) de co LPIPS + `score` cho ca 3 vung, khong
   chi PSNR/SSIM.
2. Xac nhan dinh nghia "skyline crop" voi de bai/rubric that (hien dang la
   gia dinh top-30%-chieu-cao).
3. (Tuy chon, de tower-crop dang tin hon nua) chuyen tu bounding box sang
   pixel mask that cua tru.
4. Lam lai tuong tu cho 4 scene con lai cua `public_set` neu can danh gia
   rong hon `hcm0031`.

## Neu P0 FAIL

1. Dung lai, khong chay `M0`
2. Kiem tra lai cach copy / nguon du lieu
3. Copy lai tu nguon sach khac neu can
