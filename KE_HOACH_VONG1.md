# Kế hoạch cải tiến — nhánh feature/antenna-region-focus (BTS Digital Twin NVS)

> Ghi lại từ kết quả thực chạy `bts-digital-twin-public-antenna-region-focus.ipynb` trên scene
> `hcm0031` (3DGS gốc, 30000 iter, `ENABLE_ANTENNA_FOCUS=False`) và phần thảo luận cải tiến sau đó.

## 1. Kết quả hiện tại (đã verify từ log + ảnh thật, không suy đoán)

- PSNR mean = 21.683 (min 19.248), SSIM mean = 0.6818, LPIPS mean = 0.1542, Score mean = 0.7597
  (công thức BTC mục 8.4, PSNR_max=30.0 — giả định).
- Điểm **đều trên cả 50 ảnh test**, không có outlier sập điểm → đây là giới hạn hệ thống của
  pipeline hiện tại, không phải bug ở vài pose/ảnh cụ thể.
- Ảnh so sánh (thật vs render): bố cục/màu sắc/nhà cửa khớp tốt, nhưng render **mờ nhẹ ở chi
  tiết nhỏ** (mái ngói, dây cáp, giàn cột anten) — lỗi điển hình của 3DGS thuần khi cảnh có độ
  sâu chênh lệch lớn trong 1 khung hình (drone chụp xiên: vừa thấy mái nhà gần vừa thấy nền xa)
  và có cấu trúc mảnh (dàn thép, dây cáp).
- 21.68dB thấp hơn benchmark 3DGS outdoor phổ biến (Mip-NeRF360 ~24-27dB) → còn dư địa cải
  thiện thật, chưa chạm trần.
- Quan trọng: **`ENABLE_ANTENNA_FOCUS` chưa hề được bật/test trong lần chạy này** — feature
  chính của nhánh (`07_build_antenna_weights.py` + `apply_antenna_patch.py`) vẫn chưa có số liệu
  thật để đánh giá.

## 2. Vấn đề cần lưu ý: antenna-focus có thể không giúp (thậm chí hại) điểm thi

Điểm chấm chính thức là **so khớp toàn ảnh** (PSNR/SSIM/LPIPS trên cả frame, không tách vùng
anten riêng — xem `05_eval_metrics.py`/mục 8.4 đề bài). Trong khi `apply_antenna_patch.py` làm
2 việc:

1. Nhân loss L1 cao hơn (mặc định x4) trong bbox anten — **không lấy mất ngân sách train của
   ảnh khác**, tương đối an toàn.
2. **Resample lại tần suất camera** — ảnh thấy nhiều anten được chọn train thường xuyên hơn,
   nghĩa là ảnh khác bị chọn ít hơn. Điều này có nguy cơ **giảm** PSNR/SSIM trung bình toàn ảnh,
   vì điểm thi không thưởng riêng cho độ nét vùng anten.

**Việc cần làm trước khi đầu tư thêm**: A/B test 1 scene, chạy 2 lần
(`ENABLE_ANTENNA_FOCUS=True` vs `False`), so `eval_metrics.csv` cả 2:
- Nếu Score tổng không tăng (hoặc giảm) → chỉ giữ phần loss-weighting (ý tưởng 1), bỏ phần
  resample view (ý tưởng 2).
- Nếu Score tổng tăng → giữ nguyên, tiếp tục tune `--weight`/`--margin`.

## 3. Cải tiến hợp lý cho 3DGS thuần (trước khi nghĩ đổi model)

1. **Mip-Splatting** — thay rasterizer bằng bản có 3D smoothing filter, xử lý đúng vấn đề "cảnh
   có scale/độ sâu chênh lệch lớn trong 1 ảnh" (aerial oblique). Rẻ (chỉ đổi submodule
   rasterization), đánh trúng triệu chứng mờ chi tiết đang thấy — **đáng thử nhất**.
2. **Giảm `densify_grad_threshold`** (mặc định 0.0002) hoặc kéo dài `densify_until_iter` — cấu
   trúc mảnh (dàn thép/cáp) cần mật độ Gaussian dày hơn để không bị nuốt vào background mờ.
3. **Exposure/appearance embedding** — ảnh drone chụp nhiều thời điểm trong buổi bay dễ lệch
   sáng/trắng cân bằng; 3DGS gốc không có cơ chế bù, ép model thoả hiệp độ nét để khớp màu
   trung bình → giảm SSIM/PSNR đều khắp (khớp với biểu đồ điểm rất đều đang quan sát được).
4. Vật thể chuyển động ở tầng đường (xe máy, người) trong ảnh drone tạo nhiễu "transient" — nếu
   quan sát thấy artifact rõ, nên mask hoặc bỏ qua vùng đó khi tính loss.

## 4. Nếu muốn đổi/bổ trợ model

- **2D Gaussian Splatting (2DGS)** — bề mặt phẳng (mái nhà, tường) được ràng buộc tốt hơn, giảm
  floaters, hợp cảnh đô thị nhiều mặt phẳng.
- **Scaffold-GS** — anchor phân cấp, hợp cảnh lớn có cả near/far, có thể giảm floaters tổng thể.
- **DroneSplat** — thiết kế riêng cho ảnh drone (bù exposure, xử lý vật thể chuyển động) — đúng
  bài toán hơn 3DGS gốc.
- **Nerfacto** — dùng làm model thứ 2 để ensemble/blend theo từng ảnh test (chọn/blend kết quả
  PSNR-SSIM cao nhất mỗi ảnh) thay vì chỉ dựa 1 mình 3DGS.

## 5. Thứ tự ưu tiên thực hiện

1. Bật thử `ENABLE_ANTENNA_FOCUS=True`, so Score với bản `False` hiện có — có số liệu thật thay
   vì đoán.
2. Thử Mip-Splatting làm rasterizer thay thế — rẻ, đánh đúng triệu chứng mờ chi tiết.
3. Nếu vẫn không đủ, mới tính đổi hẳn kiến trúc (2DGS/Scaffold-GS/DroneSplat) hoặc ensemble với
   Nerfacto.
