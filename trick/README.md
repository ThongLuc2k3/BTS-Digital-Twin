# Trick Workspace

Workspace này gom các mặc định để thử `trick` cho `hcm0031` mà không phải nhớ lại từng lệnh rời.

Nó bám đúng hướng trong [trick tham khảo.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick%20tham%20khảo.md):

1. `B2 dense-stereo pilot`
2. train hoặc resume với `SOURCE_MODE=prepared`
3. quét checkpoint để chọn mốc tốt nhất theo `full-image score`
4. `M0-mask` để đo đúng bệnh của tower
5. chỉ sau đó mới tuning nhẹ

## Cấu trúc

```text
trick/
  README.md
  scripts/
    01_dense_pilot.sh
    02_prepared_train.sh
    03_checkpoint_sweep.sh
  hcm0031/
    default.env
    experiment_matrix.csv
    notes_template.md
```

## Baseline đang lấy làm mốc

Theo [trick tham khảo.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick%20tham%20khảo.md):

- `full-image`: `PSNR 21.6938`, `SSIM 0.6819`, `LPIPS 0.1542`, `Score 0.6731`
- `tower-crop`: `PSNR 23.2912`, `SSIM 0.7287`, `LPIPS 0.1298`, `Score 0.7064`
- `skyline-crop`: `PSNR 20.4286`, `SSIM 0.6298`, `LPIPS 0.1829`, `Score 0.6384`

Trong workspace `trick/` này, metric tự động ưu tiên `full-image` vì đó là chỉ số thắng/thua cuối cùng.

## Cách dùng nhanh

1. Mở [trick/hcm0031/default.env](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick/hcm0031/default.env) và chỉnh:
- `GS_REPO`
- `COLMAP_BIN`
- `DATASET_ROOT` nếu không dùng mặc định

2. Chạy dense pilot:

```bash
bash trick/scripts/01_dense_pilot.sh
```

3. Nếu dense ổn, train từ `prepared`:

```bash
bash trick/scripts/02_prepared_train.sh
```

4. Quét checkpoint:

```bash
bash trick/scripts/03_checkpoint_sweep.sh
```

5. Nếu cần chẩn đoán tower đúng hơn:

```bash
bash trick/scripts/04_bootstrap_tower_masks.sh
bash trick/scripts/05_run_m0_mask_eval.sh
```

6. Ghi kết quả thật vào:
- [trick/hcm0031/experiment_matrix.csv](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick/hcm0031/experiment_matrix.csv)
- [trick/hcm0031/notes_template.md](/home/thongluc/Khóa%20Luận%20Tốt%20Nghiệp/BTS%20Digital%20Twin/trick/hcm0031/notes_template.md)

## Output chính

- Dense pilot:
  `pipeline/work/hcm0031/logs/04_colmap_dense_summary.txt`
- Prepared train:
  `pipeline/work/hcm0031/gs_model`
- Checkpoint sweep:
  `pipeline/work/hcm0031/trick_runs/checkpoint_sweep/summary.csv`
- M0-mask:
  `trick/hcm0031/m0_mask/metrics/masked_eval_summary.txt`

## Nguyên tắc dùng workspace này

- Không đánh giá `B2` bằng cảm giác nhìn ảnh.
- Mỗi lần chỉ đổi một biến chính.
- Luôn giữ log, CSV và ghi chú cho từng lần chạy.
- Không dùng `tower-crop` làm tín hiệu duy nhất nếu `tower-mask` chưa được kiểm tra.
- Nếu `prepared` không thắng `full-image score 0.6731`, dừng sớm trước khi mở thêm trick mới.
