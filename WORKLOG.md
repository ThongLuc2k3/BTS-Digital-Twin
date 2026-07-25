# WORKLOG

## 2026-07-24 — Thí nghiệm A: cách ly confound `LOW_VRAM_PROFILE` cho `B2`

### Bối cảnh
Theo `.ai-debate/01-07` và `trao đổi.md` (mục "Tổng hợp tranh luận 2026-07-24"), nhánh `B2`/`prepared` đã có failed run thật (`Score ~0.40` so với baseline `0.6731`), và nghi phạm số 1 là `LOW_VRAM_PROFILE=auto` tự động bật trên Kaggle/Colab (`RESOLUTION=4`, tắt hoàn toàn densification). `.ai-debate/07_claude_audit.md` xác định rõ: phải chạy **2 run** (2a override tích cực + 2b control âm) mới đủ bằng chứng nhân quả — chỉ chạy 1 run rồi kết luận là không đủ.

### Đã làm (không cần GPU, chuẩn bị thí nghiệm)
1. `pipeline/scripts/03_train_3dgs.sh`: sửa 1 dòng để `MODEL_DIR` tôn trọng biến môi trường override thay vì hard-code theo `SCENE` — để các run isolation không ghi đè lên `gs_model` thật (baseline hiện đã thiếu artifact, không nên phá thêm).
2. Viết `pipeline/scripts/generate_b2_isolation_notebooks.py`, sinh ra 2 notebook từ `downloads/B2_done.ipynb`:
   - `downloads/B2_isolation_2a_low_vram_off.ipynb` — `LOW_VRAM_PROFILE=0` ép tường minh, `RESOLUTION=-1`, KHÔNG dùng `--depths`.
   - `downloads/B2_isolation_2b_low_vram_control.ipynb` — `LOW_VRAM_PROFILE=1` ép tường minh (tái lập đúng cấu hình cũ), cũng KHÔNG dùng `--depths`.
   - Cả hai redirect output sang `pipeline/work/hcm0031/trick_runs/b2_isolation_<id>/` (không đụng `gs_model` thật), và có cell tự động in kết luận GO/STOP so với baseline `0.6731` và vùng fail cũ `~0.40`.
3. Cập nhật `trick/hcm0031/experiment_matrix.csv`: sửa nhãn sai `baseline_ref.source_mode` từ `raw` → `prepared` (theo bằng chứng `cfg_args` thật), đánh dấu `prepared_train_template` là `superseded` (đã có kết quả thật), thêm 2 dòng `b2_isolation_2a/2b` trạng thái `queued_gpu`.

### Next steps (cần GPU — Kaggle/Colab)
1. Chạy `downloads/B2_isolation_2a_low_vram_off.ipynb` trên Kaggle/Colab GPU (full run từ đầu: dataset setup, COLMAP CUDA build, GS repo clone, rồi train+render+eval).
2. Chạy `downloads/B2_isolation_2b_low_vram_control.ipynb` tương tự.
3. Đối chiếu 2 kết quả:
   - Nếu 2a phục hồi gần `0.6731` VÀ 2b tái lập lại vùng `~0.40` → xác nhận `LOW_VRAM_PROFILE` là nguyên nhân chính, `B2`/`prepared` (không depth) chưa bị loại, có thể tiếp tục thử `prepared + true depths` sau đó.
   - Nếu 2a vẫn sập → `LOW_VRAM_PROFILE` không phải nguyên nhân duy nhất/chính, cần điều tra tiếp render-parity hoặc bản thân prepared source trước khi thử depth.
4. Cập nhật `trick/hcm0031/experiment_matrix.csv` (điền `full_image_score`, `status=done`) và `trao đổi.md` mục 7/8 sau khi có số liệu, theo đúng khuyến nghị của `.ai-debate/07_claude_audit.md`.
5. Song song (không phụ thuộc kết quả trên, chi phí thấp — xem `.ai-debate/02_claude_review.md` mục 7): Oracle trần khả thi (geometry-assisted warp/blend, không cần train) để biết `0.85` có khả thi với 3DGS thuần hay không.
