# Tổng hợp kết quả & kế hoạch — Vòng 1 BTS Digital Twin

Nhánh `coordination/round1-status` — không gian trung tâm để theo dõi số liệu thật
từ 4 nhánh code (`main`, `feature/mip-splatting`, `compact/compact-gaussian`,
`feature/depth-anything-v2`) và chốt việc cần làm tiếp, thay vì mỗi nhánh tự ghi
chú rời rạc. Chi tiết lý luận đầy đủ xem `prompt_76diem.md` (bản trao đổi với AI
khác về chiến lược tới 76 điểm, đã qua 2 vòng phản biện).

## Mục tiêu

- Điểm cần đạt: **≥76** (Top 1 hiện ~74.9).
- Deadline: **30/07/2026**.
- Công thức chấm (đã xác nhận thật từ 1 lần BTC chấm, không phải giả định):
  ```
  Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
  PSNR_norm = clamp(PSNR / 50, 0, 1)
  ```
  Độ nhạy quy đổi: −0.01 LPIPS = +0.4đ | +0.01 SSIM = +0.3đ | +1dB PSNR = +0.6đ.

## Baseline đã chấm thật

Vanilla 3DGS (`main`, không sửa gì): **Score = 58.67320**
(PSNR 19.471466, SSIM 0.563734, LPIPS 0.248042), trung bình 8 scene `private_set1`.

## 3 nguyên nhân gốc đã xác định (không suy đoán, đối chiếu ảnh thật)

- **(a)** Cấu trúc mảnh (ăng-ten/cáp/khung thép) → Gaussian phải phình to để phủ → mờ/răng cưa.
- **(b)** Pose test xa camera train → dễ sinh floaters ở vùng khuyết.
- **(c)** Nền (trời/mây/cây) chiếm phần lớn khung hình → ngân sách Gaussian lãng phí, không dồn cho ăng-ten.

## Trạng thái từng nhánh

| Nhánh | Kỹ thuật | Đánh vào nguyên nhân | Trạng thái | Số liệu |
|---|---|---|---|---|
| `main` | Vanilla 3DGS | — | Đã chấm thật (private, 8 scene) | Score 58.67 |
| `feature/mip-splatting` | Antialiasing (Mip-Splatting EWA) | (a) | Đã vá bug train/render lệch config, chạy 1 scene public | `hcm0031`: PSNR 21.69, SSIM 0.682, LPIPS 0.158, **Score 67.15** (⚠️ 1 scene, chưa đại diện 8 scene private) |
| `feature/mip-splatting` | + Depth-prior (Depth Anything V2) | (b) | Code sẵn, **chưa chạy** | — |
| `feature/mip-splatting` | + Exposure comp (`--train_test_exp`) | lệch sáng/màu | Code sẵn, **chưa chạy**; đã xác nhận qua source: KHÔNG ảnh hưởng ảnh nộp bài (chỉ giúp hội tụ lúc train), pose test mới luôn render không áp exposure | — |
| `compact/compact-gaussian` | Gaussian Volume Mask (nén) + antenna-box bảo vệ | (c) | Code xong, **CHƯA chạy GPU thật** | — |
| `feature/depth-anything-v2` | Depth Anything V2 depth-prior | (b) | Từng CUDA OOM ở iter ~11100/30000 (T4 14.5GB). Đã tăng `densify_grad_threshold` 0.0002→0.0004 né OOM | **Chưa có lần chạy sạch sau khi né OOM** |

## Công cụ đã có sẵn, chưa dùng hết

- `pipeline/scripts/09_diagnose_distance.py` — tương quan PSNR/SSIM/LPIPS vs khoảng
  cách camera train gần nhất. Đã có trên `main` và `feature/mip-splatting`.
  ⚠️ Điểm yếu đã tự nhận ra: khoảng cách Euclid là proxy yếu cho độ khó — nên cân
  nhắc đo thêm góc lệch hướng nhìn (viewing angle) nếu muốn kỹ hơn.

## Việc cần làm tiếp theo (đã thống nhất qua phản biện, theo đúng thứ tự)

1. **[x] Kiểm tra độ phân giải ảnh** — ĐÃ KIỂM TRA (đọc trực tiếp `test_poses.csv`
   + ảnh train của cả 13/13 scene, public lẫn private): tất cả đồng nhất **1320×989**,
   thấp hơn hẳn ngưỡng auto-downscale 1600px của repo Inria gốc. **Không có rò rỉ
   điểm số ở khâu này — loại bỏ giả thuyết, không cần sửa gì.**
2. **[ ] Sanity-check train/render nhất quán** trên `feature/mip-splatting` — render
   lại vài ảnh TRAIN qua đúng đường `04_render_test_poses.py`, so PSNR với log lúc
   train (`03_train_3dgs.sh` in ra lúc training_report). Lệch >0.1dB = có mismatch
   ẩn khác chưa phát hiện (bài học từ bug antialiasing 10 điểm).
3. **[ ] Chạy đủ 4-5 scene public trên `feature/mip-splatting`** (chỉ antialiasing,
   giữ nguyên cấu hình đã dùng cho `hcm0031`) — lấy số liệu trung bình đáng tin
   thay vì suy đoán từ 1 scene. Đây là bước TẤT CẢ các phân tích đều đồng ý là ưu
   tiên #1 trước khi quyết định bất kỳ hướng đi tiếp theo nào.
4. **[ ] Hoàn thiện `compact/compact-gaussian`**: chạy thật ít nhất 1 scene để có
   số liệu, dùng để đối chiếu với quyết định "có nên bỏ hướng pruning-mask hay
   không" (AI phản biện đề xuất bỏ, nhưng dựa trên suy luận, chưa có số liệu thật
   của chính dataset này để xác nhận).
5. **[ ] Hoàn thiện `feature/depth-anything-v2`**: chạy lại sạch sau khi né OOM,
   lấy số liệu thật để so trực tiếp với antialiasing-only.
6. **[ ] Cân nhắc nhánh 6 (gsplat + 3DGS-MCMC)** — CHỈ sau khi có số liệu thật từ
   bước 3-5, và chỉ nếu khoảng cách còn lại tới 76 vẫn lớn. Không vội pivot khi
   4 nhánh hiện có còn chưa được đo đạc đầy đủ. Nếu làm: 1 người phụ trách riêng,
   có mốc go/no-go rõ ràng (vd tới ngày X phải render+đóng gói submission được,
   vượt antialiasing-only ≥0.5 điểm mới giữ).

## Câu hỏi còn mở (chưa có câu trả lời chắc chắn)

- Trung bình 8 scene private nếu chỉ dùng antialiasing sẽ rơi vào khoảng nào?
  (67.15 chỉ là 1 scene, có thể thuận lợi hơn mặt bằng chung).
- Cộng dồn depth-prior + antialiasing có cộng dồn tuyến tính hay đã có phần chồng
  lấn (antialiasing đã giải quyết 1 phần floaters/răng cưa rồi)?
- Ảnh BTC gốc có vượt 1600px không? (việc 1, chưa kiểm tra).
