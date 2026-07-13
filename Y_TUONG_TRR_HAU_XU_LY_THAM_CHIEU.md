# Ý tưởng mới: TRR — Tiered Reference-guided Refinement (Hậu xử lý tham chiếu theo tầng)

> Ghi lại bối cảnh + toàn bộ trao đổi dẫn tới ý tưởng này, để không phải giải thích lại
> từ đầu ở phiên làm việc sau. Xem `Hướng đi.md` để biết kế hoạch chính đang chạy — file
> này chỉ mô tả **một hướng mở rộng mới**, chưa triển khai.

## 0. Bối cảnh (tóm tắt trạng thái dự án tại thời điểm nảy ý tưởng)

- Bài toán: từ 150–300 ảnh drone/scene (`train/images/`), render ảnh RGB tại pose BTC chỉ
  định cho 8 scene `private_set1`, chấm bằng `Score = 0.4×(1−LPIPS) + 0.3×SSIM +
  0.3×PSNR_norm` (điểm trung bình theo scene). Đã nộp thật, đạt **58.67320**
  (PSNR 19.47, SSIM 0.5637, LPIPS 0.2480) bằng vanilla 3DGS gốc — Top 1 hiện ~74.9.
- Pipeline hiện tại (`pipeline/scripts/`) đã có: COLMAP pose có sẵn từ BTC, train
  3DGS/gsplat-MCMC, Mip-Splatting (`--antialiasing`), Depth Anything V2 prior, exposure
  compensation, antenna-region-focus, fine-tune LPIPS (`03b_finetune_lpips_gsplat.py`).
  Toàn bộ các hướng này tác động vào **giai đoạn train** — sửa cách Gaussian được tối ưu.
- Nguyên nhân điểm bị kẹt (đã phân tích ở `Hướng đi.md` mục 2): (a) cấu trúc mảnh
  (ăng-ten/dây cáp) khiến Gaussian phải phình to → mờ; (b) pose test ở góc không có ảnh
  train → floaters vùng khuyết; (c) nền trời/mây/cây chiếm phần lớn khung hình, loss chia
  đều lãng phí Gaussian vào nền.

## 1. Câu hỏi khởi điểm của user

> "tôi muốn có thêm 1 mô hình nữa dựa vào kết quả vừa tạo ra và file ảnh đã dùng để train
> để lấy đó đối chiếu và sửa bổ sung làm rõ làm nét lên kết quả — có model nào không hoặc
> chúng ta sẽ tự làm?"

Diễn giải lại: thêm **một giai đoạn thứ 2, sau khi 3DGS đã render xong** ảnh ở pose test —
không sửa cách train, mà lấy (a) ảnh vừa render + (b) ảnh train thật gần nhất (đã biết pose
qua COLMAP) làm input cho một model/quy trình khác, để **đối chiếu phát hiện chỗ render bị
mờ/thiếu chi tiết** rồi bổ sung/làm nét dựa trên texture thật đã thấy trong ảnh train.

Đây là bài toán **reference-guided post-render refinement**, khác hẳn về bản chất so với
5 hướng đang làm (vốn đều là "sửa lúc train"). Đã có literature/model có sẵn cho việc này,
không phải tự nghĩ ra từ đầu.

## 2. Các lựa chọn đã khảo sát

| # | Hướng | Cách hoạt động | Cần train model? |
|---|---|---|---|
| A | **Geometric reprojection / MVS texture blending** | Dùng chính point cloud + pose COLMAP đã có: với mỗi pixel ảnh render, tìm ảnh train nào nhìn thấy điểm 3D đó gần nhất, warp/project patch texture thật của ảnh train đó lên đúng vị trí, blend theo trọng số góc nhìn | **Không** — thuần hình học |
| B | **3DGS-Enhancer** (NeurIPS 2024, `chen-yingjie/3DGS-Enhancer`) | Video-diffusion prior train sẵn chuyên sửa artifact 3DGS render ở novel-view thưa (blur, inconsistency), có điều kiện theo ảnh tham chiếu | Dùng pretrained, có thể fine-tune |
| C | **RefSR (Reference-based Super-Resolution)** — MASA-SR, C2-Matching, AMSA | Match patch/feature giữa ảnh chất lượng thấp và ảnh tham chiếu chất lượng cao, transfer texture | Dùng pretrained hoặc train nhẹ |
| D | **Diffusion img2img + IP-Adapter/ControlNet-Tile tham chiếu ảnh train** | SDEdit denoise-thấp trên ảnh render, "mồi" bằng ảnh train qua IP-Adapter | Dùng pretrained, không train riêng |

## 3. Đánh giá khả thi

**Rủi ro cốt lõi (áp dụng cho B, C, D):** điểm số chấm bằng LPIPS/SSIM/PSNR **so với ảnh
GT thật**, không phải "nhìn đẹp mắt hơn". Diffusion/RefSR có thể "bịa" chi tiết hợp lý về
mặt hình ảnh nhưng sai lệch với cấu trúc cơ khí thật (ăng-ten, dây cáp) ở đúng pose đó —
nếu bịa sai, LPIPS thậm chí **tệ hơn** dù ảnh trông nét hơn bằng mắt thường. Đây cũng chính
là lý do mục #6 trong `Hướng đi.md` xếp diffusion-based view completion là "rủi ro cao,
effort cao" — B/D nằm chung nhóm rủi ro đó.

**Hướng A khác hẳn về bản chất rủi ro:** texture lấy đúng từ ảnh train thật (biết chính
xác điểm 3D nào nhìn thấy ở đâu qua sparse COLMAP có sẵn), không "bịa" — chỉ chọn lọc/blend
lại thông tin đã tồn tại. Rủi ro thấp hơn nhiều, không cần GPU mạnh, không vướng luật thi
(mục 4 `Hướng đi.md` — không cần clone/pin thêm pretrained weight mới), tận dụng được hạ
tầng pose/sparse đã có (`pipeline/common/poses.py`, `alignment.py`).

**Hạn chế của A:** chỉ sửa được vùng **có visibility tốt** từ ảnh train (điểm 3D đã được
quan sát ít nhất 1 ảnh train). Vùng hoàn toàn khuất ở mọi ảnh train (đúng nguyên nhân (b)
— floaters vùng khuyết) thì A bó tay, phải quay lại Depth Anything V2 (#3) hoặc diffusion
completion (#6) đã có trong kế hoạch chính.

## 4. Ý tưởng kết hợp: **TRR — Tiered Reference-guided Refinement**

Đặt tên cho hướng đi mới, kết hợp A + B thành pipeline hậu xử lý **2 tầng**, chạy sau khi
`04_render_gsplat_test_poses.py` đã render xong ảnh ở pose test:

- **Tầng 1 (bắt buộc, an toàn):** geometric reprojection (hướng A) — với mọi pixel render,
  nếu điểm 3D tương ứng có visibility tốt từ ≥1 ảnh train, project/blend texture thật vào
  để làm nét lại. Không hallucinate, không cần model mới.
- **Tầng 2 (tuỳ chọn, chỉ bật nếu Tầng 1 đo có lợi và còn dư effort/GPU):** với các pixel
  Tầng 1 không sửa được (vùng khuất hoàn toàn khỏi mọi ảnh train), dùng 3DGS-Enhancer
  (hướng B) — model chuyên biệt cho đúng bài toán "sửa artifact 3DGS render thưa view",
  rủi ro hallucination thấp hơn diffusion tổng quát (D) vì được train riêng cho tác vụ này.

Không đưa C/D vào TRR — rủi ro hallucinate cao nhất so với lợi ích, không phù hợp với cách
chấm điểm dựa trên khoảng cách tới GT thật.

## 5. Đề xuất triển khai (nếu quyết định làm)

1. Code Tầng 1 (hướng A) trước — effort thấp, rủi ro thấp, đo thử ngay trên `public_set`
   (có ảnh GT thật để so PSNR/SSIM/LPIPS) trước khi đụng gì khác.
2. Chỉ cân nhắc Tầng 2 (3DGS-Enhancer) nếu Tầng 1 đo có cải thiện rõ trên `public_set` và
   còn dư thời gian/GPU sau khi hoàn tất #1–5 trong `Hướng đi.md`.
3. Coi TRR là **nhánh mở rộng của mục #6 "dự phòng"** trong `Hướng đi.md` (hậu xử lý sau
   render, khác với augment lúc train) — không chen ngang thứ tự ưu tiên 1→5 đã chốt.

## 6. Trạng thái

Mới dừng ở mức ý tưởng/đánh giá khả thi — **chưa code, chưa chạy thử**. Việc cần làm tiếp
theo nếu quyết định theo hướng này: thiết kế script Tầng 1 (đọc sparse COLMAP + pose test,
tìm ảnh train nhìn thấy điểm 3D gần pixel render, warp + blend), thêm vào pipeline sau bước
04, test trên 2–3 scene `public_set` so với baseline không hậu xử lý.
