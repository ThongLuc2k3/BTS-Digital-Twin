# M0-Mask Workflow

`M0-mask` là nhánh chẩn đoán cho phần tower khi `tower-crop` kiểu bounding box không còn đủ tin cậy.

Mục tiêu:

- chấm metric chỉ trên pixel thuộc tower
- biết `tower-crop` đang bị kéo điểm lên bao nhiêu bởi pixel nền
- tạo tín hiệu đúng hơn để quyết định có nên làm local refine cho tower hay không

## Ý tưởng

Workflow này chia làm 2 lớp:

1. `bootstrap mask`
- chiếu `tower_bbox3d.json` lên từng ảnh test
- tạo polygon mask thô để làm điểm khởi đầu

2. `manual-correct + masked eval`
- sửa lại mask thô cho ôm sát tower/anten/wire
- chấm lại metric chỉ trên vùng mask

## Cấu trúc được dùng

```text
trick/hcm0031/m0_mask/
  bootstrap_masks/
  manual_masks/
  metrics/
  review/
```

## Quy ước file

- Tên mask khớp với tên render/GT theo `stem`
- Ảnh gốc: `DJI_..._V.JPG`
- Render: `DJI_..._V.png`
- Mask: `DJI_..._V.png`

Mask phải là ảnh nhị phân:

- pixel `>0` được coi là thuộc tower
- pixel `0` là ngoài tower

## Chạy nhanh

1. Tạo mask thô:

```bash
bash trick/scripts/04_bootstrap_tower_masks.sh
```

2. Sửa tay các mask trong:
- `trick/hcm0031/m0_mask/bootstrap_masks`

3. Lưu bản đã sửa vào:
- `trick/hcm0031/m0_mask/manual_masks`

4. Chấm masked metric:

```bash
bash trick/scripts/05_run_m0_mask_eval.sh
```

## Output quan trọng

- `bootstrap_masks/*.png`: mask thô từ OBB
- `metrics/masked_eval.csv`: metric từng ảnh
- `metrics/masked_eval_summary.txt`: trung bình và coverage

## Cách diễn giải

- Nếu `tower-mask score` thấp hơn `tower-crop score` nhiều:
  crop cũ đang bị nhiễm nền nặng, chưa được dùng để chọn tower trick.
- Nếu `tower-mask score` cũng vẫn khá ổn:
  tower không phải nút nghẽn chính, ưu tiên background trước.
- Nếu score tower thấp tập trung ở vài view:
  nên nghĩ đến hard-view refine thay vì đổi toàn bộ pipeline.
