# Bối cảnh

Tôi đang thi **VIETTEL AI RACE 2026 — đề BTS Digital Twin (Novel View Synthesis)**,
vòng 1 (vòng sơ loại). Deadline nộp bài: **30/07/2026**. Tôi cần một **kế hoạch kỹ
thuật cụ thể, có thứ tự ưu tiên** để đưa điểm số từ hiện trạng lên **tối thiểu 76
điểm** (leaderboard Top 1 hiện ~74.9, tôi cần vượt qua).

## Luật chấm điểm (chính xác, không suy đoán)

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)
```

`PSNR_max` không được BTC công bố công khai, nhưng tôi đã **giải ngược được từ 1 lần
chấm thật**: nộp bản vanilla 3DGS (không sửa gì) trên `private_set1` (8/8 scene khớp),
BTC chấm **Score = 58.67320** với PSNR=19.471466, SSIM=0.563734, LPIPS=0.248042 →
giải ra `PSNR_max ≈ 50.0000` (sai số <0.001%). Dùng con số này làm chuẩn từ giờ.

## Dữ liệu & hạ tầng

- 13 scene: 5 scene `public_set` (có ảnh ground-truth để tự chấm PSNR/SSIM/LPIPS),
  8 scene `private_set1` (không có ảnh thật, chỉ BTC chấm được). Mỗi scene: ảnh drone
  chụp trạm BTS (cột ăng-ten viễn thông), 150-300 ảnh train/scene, 40-70 pose test/scene.
- Điểm cuối cùng = **trung bình theo SCENE** (không theo ảnh) trên 8 scene private.
- Hạ tầng tôi tự lo: đang dùng Kaggle (GPU T4, ~14.5GB VRAM khả dụng, ~30 giờ
  GPU/tuần, tối đa ~12 giờ/phiên) và/hoặc Colab free tier. BTC gợi ý cấu hình tham
  khảo cho inference là 1× RTX A4000 20GB — tôi CHƯA có GPU mạnh hơn T4/P100.
- Đội 3 người. Còn khoảng 20 ngày tới deadline.

## Baseline đã chấm thật (mốc xuất phát)

Vanilla 3DGS (`graphdeco-inria/gaussian-splatting` gốc, không sửa gì, dùng thẳng
sparse COLMAP do BTC cung cấp): **Score thật = 58.67** (PSNR 19.47, SSIM 0.564,
LPIPS 0.248), trung bình 8 scene private.

## 3 nguyên nhân gốc đã xác định khiến baseline thấp (phân tích ảnh thật, không suy đoán)

- **(a)** Trạm BTS có cấu trúc mảnh (ăng-ten, dây cáp, khung thép) — Gaussian
  ellipsoid phải phình to để phủ đủ → mờ/răng cưa ở đúng vùng BTC chấm trọng tâm.
- **(b)** Pose test đôi khi nằm ở góc không có ảnh train gần → dễ sinh floaters
  (vệt Gaussian lơ lửng sai) ở vùng khuyết.
- **(c)** Nền trời/mây/cây/nhà xung quanh chiếm phần lớn khung hình → ngân sách
  Gaussian bị "lãng phí" vẽ nền thay vì tập trung vào ăng-ten.

## Hiện trạng 4 nhánh code tôi đang làm song song

1. **`main`** — baseline vanilla nguyên bản, chính là bản đã chấm 58.67 ở trên.

2. **`feature/mip-splatting`** — bật `--antialiasing` (EWA Filter của Mip-Splatting,
   đã tích hợp sẵn trong repo Inria từ bản 10/2024, không cần đổi rasterizer) — đánh
   vào nguyên nhân (a). Vừa sửa xong 1 bug nghiêm trọng (rasterizer render lệch cấu
   hình so với lúc train, làm điểm số bị đánh giá SAI THẤP suốt nhiều lần chạy trước
   đó). Sau khi vá: chạy 1 scene public (`hcm0031`) chỉ với antialiasing (CHƯA bật
   depth-prior) → **PSNR=21.69, SSIM=0.682, LPIPS=0.158, Score=67.15** — cao hơn hẳn
   baseline 58.67, nhưng **đây chỉ là 1 scene public, chưa đại diện cho trung bình
   8 scene private thật**. Nhánh này cũng có sẵn (nhưng CHƯA chạy thật):
   depth-prior (Depth Anything V2, đánh vào nguyên nhân b) và exposure compensation
   (`--train_test_exp`, đánh vào lệch sáng/màu giữa các ảnh train, nếu có).

3. **`compact/compact-gaussian`** — cài "Compact 3D Gaussian Representation"
   (Lee et al., arXiv:2311.13681, mục 3.1 "Gaussian Volume Mask") — học 1 mask nhị
   phân để cắt tỉa Gaussian dư thừa, giảm số Gaussian mà cố giữ chất lượng. Có thêm
   cơ chế khai báo hộp bao 3D quanh ăng-ten/RRU/cáp để BẢO VỆ các Gaussian đó khỏi bị
   cắt tỉa (đánh vào nguyên nhân c — giải phóng ngân sách Gaussian khỏi nền, dồn cho
   ăng-ten). **CHƯA chạy thật trên GPU** để lấy số liệu — mới rà soát code tĩnh.

4. **`feature/depth-anything-v2`** — dùng Depth Anything V2 sinh depth map cho ảnh
   train, bật depth regularization (`--depths`) của repo gốc — đánh vào nguyên nhân
   (b). Từng bị **CUDA out of memory** giữa chừng (train tới iteration ~11100/30000
   thì hết 14.5GB VRAM, do densify sinh Gaussian quá nhanh — đặc thù scene BTS nhiều
   chi tiết mảnh). Đã tăng `densify_grad_threshold` từ 0.0002 lên 0.0004 để né OOM
   (đánh đổi: ít Gaussian hơn, có thể ảnh hưởng đúng vùng ăng-ten/cáp cần nhiều
   Gaussian nhất). **CHƯA có kết quả sạch (chưa chạy lại thành công sau khi né OOM).**

## Công cụ chẩn đoán vừa xây (chưa có số liệu)

Script mới đọc lại điểm PSNR/SSIM/LPIPS từng ảnh test + khoảng cách từ ảnh test đó
tới camera train GẦN NHẤT, tính tương quan Pearson — mục đích xác định nguyên nhân
(b) có thực sự là nút thắt chính ở từng scene hay không, trước khi quyết định có
đáng đẩy mạnh depth-prior hay không. Chưa chạy để có số liệu thật.

## Băn khoăn cụ thể cần AI giúp phân tích

Kết quả tốt nhất hiện có (67.15) chỉ đến từ **1 kỹ thuật** (antialiasing) trên
**1 scene public** — không phải trung bình 8 scene private thật. Cải thiện +8.5
điểm này gần như chắc chắn là do sửa lỗi (không phải do kỹ thuật mạnh), nên tôi
KHÔNG chắc trung bình 8 scene private (nếu chỉ dùng antialiasing) có đạt nổi mức
đó không, và càng không chắc cộng dồn thêm depth-prior + compact-gaussian +
antenna-focus có đủ sức kéo từ ~65-68 lên 76 hay không — đây là suy đoán định
tính từ tài liệu nội bộ của tôi, CHƯA kiểm chứng bằng số liệu thật.

# Yêu cầu

Dựa trên toàn bộ bối cảnh trên (đặc biệt: 3 nguyên nhân gốc a/b/c, giới hạn VRAM
14.5GB, ~20 ngày còn lại, đội 3 người, và việc antialiasing một mình mới đạt 67
trên 1 scene thuận lợi), hãy:

1. **Đánh giá thực tế** khả năng đạt 76 điểm có khả thi không với các kỹ thuật/tài
   nguyên hiện có, hay cần hướng đi khác hẳn (vd đổi hẳn kiến trúc biểu diễn 3D thay
   vì vá thêm lên nền 3DGS vanilla).
2. Cho **thứ tự ưu tiên cụ thể** nên làm tiếp theo trong số: (a) chạy đủ 4-5 scene
   public để có số liệu trung bình đáng tin hơn thay vì suy đoán từ 1 scene; (b) hoàn
   thiện depth-prior; (c) hoàn thiện compact-gaussian + antenna-focus kết hợp; (d)
   thử kiến trúc khác (2DGS/Scaffold-GS) cho cấu trúc mảnh; (e) thuê GPU lớn hơn để
   bỏ giới hạn `densify_grad_threshold` đã phải tăng lên để né OOM. Với mỗi lựa chọn,
   ước lượng **định tính** mức độ có thể cải thiện điểm số và effort/rủi ro tương ứng.
3. Chỉ ra nếu có **kỹ thuật/hướng đi nào tôi đang bỏ sót** mà cộng đồng NVS/3DGS
   (2024-2026) đã chứng minh hiệu quả cho đúng bài toán: cảnh ngoài trời, cấu trúc
   nhân tạo mảnh (cột thép/ăng-ten/dây cáp), ít ảnh train (150-300 ảnh/scene), cần
   render tại pose mới hoàn toàn không có trong tập train.
4. Đề xuất cách **kiểm chứng nhanh, tốn ít GPU-giờ nhất** để biết sớm hướng nào có
   khả năng chạm 76 trước khi đổ hết 20 ngày còn lại vào 1 hướng sai.

Trả lời dạng kế hoạch có thứ tự ưu tiên rõ ràng, không cần rào trước đón sau, ưu
tiên tính thực dụng (có thể làm được với GPU T4/P100 + 3 người + 20 ngày).



Câu trả lời:

Đánh giá thẳng: 76 điểm nghĩa là gì về mặt metric
Quy đổi độ nhạy từ công thức chấm (PSNR_max = 50):

−0.01 LPIPS = +0.4 điểm (đòn bẩy lớn nhất)
+0.01 SSIM = +0.3 điểm
+1 dB PSNR = +0.6 điểm

Để chạm 76 trung bình private, bạn cần đại loại: LPIPS ≈ 0.09–0.10, SSIM ≈ 0.79–0.80, PSNR ≈ 26 dB. So với kết quả tốt nhất hiện có (0.158 / 0.682 / 21.7 — trên scene public thuận lợi), tức là còn thiếu ~6 dB PSNR, ~0.10 SSIM, ~0.06 LPIPS trên trung bình 8 scene khó hơn.
Kết luận thực tế: 76 là khả thi nhưng là kịch bản "mọi thứ đều ăn"; kịch bản trung tâm nếu bạn stack đúng kỹ thuật là ~70–74. Việc Top 1 đang ở 74.9 chứng tỏ mức đó đạt được bằng 3DGS cải tiến — không cần đổi kiến trúc hẳn. Vá lên nền 3DGS là đúng hướng, nhưng vá sai chỗ (chi tiết bên dưới): nhánh compact-gaussian và ý định thuê GPU đang giải quyết một vấn đề mà cộng đồng đã giải xong bằng cách khác rẻ hơn.
Kỹ thuật bạn đang bỏ sót (quan trọng nhất trước)
1. gsplat + MCMC densification (3DGS-MCMC, Kheradmand et al. 2024). Đây là lỗ hổng lớn nhất trong kế hoạch. MCMC thay toàn bộ cơ chế densify/prune bằng relocation với số Gaussian cố định do bạn đặt (--max-gaussians). Hệ quả trực tiếp: (i) hết vĩnh viễn OOM — không cần tăng densify_grad_threshold (cái hack đang giết đúng vùng ăng-ten cần Gaussian nhất), không cần thuê GPU; (ii) Gaussian "chết" ở nền trời được tự động di dời vào vùng lỗi cao (ăng-ten, cáp) — chính là mục tiêu của nhánh compact-gaussian + antenna-box, nhưng tự động, đã được chứng minh, không cần khai báo hộp bao thủ công cho 13 scene. gsplat có sẵn simple_trainer.py đọc thẳng COLMAP format, kèm cờ --antialiased (tương đương Mip-Splatting 2D filter) và strategy=mcmc. Với 14.5GB, đặt budget ~1.5–2.5M Gaussians.
2. Fine-tune bằng LPIPS loss ở 3–5k iterations cuối. Trọng số 0.4 cho LPIPS mà loss train mặc định (L1 + 0.2·D-SSIM) không hề tối ưu perceptual. Trick chuẩn trong các NVS challenge: sau 25–27k iters, thêm term LPIPS (VGG) tính trên random crop 256–384px (để vừa VRAM T4). Gain điển hình 0.02–0.05 LPIPS = +0.8–2 điểm, gần như miễn phí về GPU-giờ. Đồng thời thử tăng lambda_dssim 0.2 → 0.3–0.4 (SSIM chiếm 0.3 điểm số) — thử nghiệm 1 lần chạy là biết.
3. Bilateral grid / appearance compensation cho ảnh drone. Ảnh drone chụp vòng quanh trạm gần như chắc chắn lệch exposure/white balance theo hướng nắng → 3DGS học ra màu "trung bình lem". gsplat có --use-bilateral-grid. Lưu ý: --train_test_exp của repo Inria không dùng được cho bạn — nó fit exposure trên nửa trái ảnh test, mà bạn không có GT test. Bỏ nhánh đó.
4. Sky masking. Segment trời (SegFormer/SAM chạy offline trên CPU/GPU rẻ), train với sky mask → Gaussian không phí vào mây trôi (mây còn di chuyển giữa các ảnh — nguồn floaters kinh điển). Đánh trực tiếp nguyên nhân (b) và (c).
5. Kiểm tra độ phân giải — đòn bẩy im lặng. Repo Inria mặc định downscale ảnh về chiều rộng ≤1.6K. Nếu BTC chấm ở độ phân giải gốc còn bạn train/render ở 1.6K rồi upscale, bạn đang mất điểm PSNR/SSIM vô hình. Xác nhận độ phân giải BTC render/chấm và train đúng res đó (hoặc render đúng res nộp).
Cảnh báo: đừng chạy lại COLMAP hay bật pose-optimization cho train poses — test poses được định nghĩa trong hệ quy chiếu COLMAP của BTC; tinh chỉnh train poses sẽ làm lệch khung với test poses.
Thứ tự ưu tiên 20 ngày
Phase 0 (ngày 1–3): đo trước, tin sau — lựa chọn (a), làm ngay.
Chạy nhánh mip-splatting (đã vá bug) trên cả 5 scene public (~6–10 GPU-h). Chạy script chẩn đoán tương quan distance-quality trên kết quả. Nếu quota nộp cho phép, nộp luôn bản AA full 8 scene private — đây là lần calibrate public↔private duy nhất bạn có, giá trị thông tin cực lớn. Effort thấp, rủi ro ~0.
Phase 1 (ngày 3–9): cú đặt cược chính — gsplat MCMC.
Người 1: port sang gsplat, config mcmc + antialiased + bilateral-grid, budget 2M, chạy A/B trên 2 scene dev (1 scene tốt nhất + 1 tệ nhất từ Phase 0). Kỳ vọng +1.5–3 điểm so với AA-only, cộng thêm việc gỡ hoàn toàn ràng buộc OOM. Người 2: LPIPS fine-tune + quét lambda_dssim. Người 3: sky masking pipeline + xác minh độ phân giải. Depth-prior (b): chỉ chạy nếu diagnostic Phase 0 cho tương quan âm rõ (r < −0.4) ở ≥2 scene; code đã sẵn nên effort thấp, kỳ vọng +0.3–1.5 điểm tùy scene, nhưng đừng ưu tiên trước MCMC.
Phase 2 (ngày 9–14): stack và calibrate lần 2.
Gộp các cờ thắng cuộc, chạy đủ 5 public. Nếu trung bình public ≥ mức tương ứng 76 (dùng gap đo ở Phase 0 để quy đổi), chạy full 13 scene và nộp.
Phase 3 (ngày 14–20): per-scene tuning + buffer.
Điểm cuối là trung bình theo scene → 1 scene tệ kéo cả đội. Dồn budget vào 1–2 scene tệ nhất: tăng Gaussian budget riêng, thêm iters, depth-prior riêng cho scene khuyết góc. Giữ 2–3 ngày cuối cho chạy render + nộp. Thuê GPU (e): không phải để fix OOM (MCMC đã fix), nhưng ~$20–30 trên vast.ai (RTX 3090/A4000) cho tuần cuối để chạy full sweep 13 scene trong 1 đêm thay vì ăn hết quota Kaggle là khoản chi hợp lý. Tận dụng 3 tài khoản Kaggle của 3 người = ~90 GPU-h/tuần chạy song song.
Về các lựa chọn còn lại
(c) compact-gaussian + antenna-box: khuyên bỏ. Pruning theo mask chủ yếu tiết kiệm bộ nhớ chứ không tăng chất lượng (paper gốc báo chất lượng xấp xỉ, không hơn); khai báo hộp bao 3D thủ công cho 13 scene là chi phí nhân công lớn cho gain không rõ. MCMC đạt cùng mục tiêu (dồn ngân sách vào ăng-ten) tự động.
(d) 2DGS/Scaffold-GS: chỉ là contingency. 2DGS thường giảm PSNR trên metric ảnh (nó tối ưu geometry, không phải NVS score). Scaffold-GS có gain thật trên cảnh outdoor lớn nhưng chi phí tích hợp + debug trong 20 ngày là rủi ro cao. Chỉ đụng đến nếu Phase 1 plateau dưới ~70 trên public.
Giao thức kiểm chứng rẻ (câu 4)
Cố định 2 scene dev (1 tốt, 1 tệ), chạy 15k iterations thay vì 30k cho mọi A/B — thứ hạng giữa các config ở 15k gần như bảo toàn ở 30k, chi phí ~0.7–1 GPU-h/run trên T4. Cùng seed, cùng res. Ngưỡng quyết định: config mới phải ≥ +0.5 điểm trên cả 2 scene mới được vào stack; dưới ngưỡng thì bỏ, không tiếc. Quy mọi kết quả về điểm bằng bảng độ nhạy ở đầu (−0.01 LPIPS = +0.4đ...) để so sánh táo với táo. Với kỷ luật này, mỗi giả thuyết tốn <2 GPU-h và bạn biết hướng nào có cửa chạm 76 trước ngày thứ 10.



Trả lời:
Sáu câu này chất vấn chính câu trả lời của tôi, nên trả lời luôn tại đây thay vì "gửi cho AI kia". Nói trước kết luận: bạn đúng ở 1, 4, 5, đúng một nửa ở 6; câu 2 và 3 kiểm chứng được bằng source + 1 lần chạy khói, tôi trả lời phần đọc-source được, phần còn lại bạn phải chạy thật.
1. Đúng — tôi đã gộp sai 2 cơ chế và khẳng định quá tay về MCMC. Cơ chế relocation của 3DGS-MCMC chọn đích di dời theo opacity (Gaussian "chết" được chuyển tới vị trí Gaussian sống, xác suất tỷ lệ opacity), tức tín hiệu là photometric loss toàn ảnh — không có khái niệm ngữ nghĩa "ăng-ten". Mây trôi giữa các ảnh tạo inconsistency photometric → hoàn toàn có thể hút mass như bạn nói; sky masking (đã đề xuất) là thứ bịt cái sink đó, và nó trở thành điều kiện đi kèm của MCMC chứ không phải tùy chọn. Về antenna-region loss-reweighting: bạn đúng rằng nó khác pruning-mask và bổ sung được cho MCMC. Quan trọng hơn: nó rẻ hơn nhiều so với hộp bao 3D — chỉ cần mask 2D trên ảnh train (bạn có sẵn kinh nghiệm YOLOv11 từ luận văn để detect vùng ăng-ten/RRU), nhân weight map vào L1, ~20 dòng code, không phụ thuộc codebase. Đưa vào danh sách A/B với ngưỡng +0.5 điểm như giao thức cũ. Cái tôi vẫn giữ nguyên: nhánh pruning-mask (compact-gaussian) bỏ, vì mục tiêu tiết kiệm ngân sách của nó bị MCMC thay thế, còn mục tiêu chất lượng thì paper gốc cũng không hứa.
2. (Câu bị cắt cụt — tôi đoán ý là "1.5–2.5M có vừa 14.5GB không".) Toán thô: mỗi Gaussian SH bậc 3 ≈ 59 float tham số + 2 state Adam ≈ ~0.7KB → 2M ≈ 1.4GB cho tham số + optimizer. Phần ăn VRAM thật là activation lúc rasterize, phụ thuộc độ phân giải ảnh của bạn — cái này không tính chay được, phải đo. Cách đo mất 1 giờ: chạy gsplat MCMC budget 1M ở đúng res, đọc peak memory, ngoại suy tuyến tính theo budget. Con số 1.5–2.5M tôi đưa là heuristic từ các báo cáo chạy gsplat trên GPU 16GB, không phải bảo chứng cho scene và res của bạn. Nếu ý câu 2 là chuyện khác, nói lại giúp tôi.
3. Không, tôi không dám bảo là bug-compatible — và điểm mấu chốt là bạn không cần nó bug-compatible. Bài học 10-điểm của bạn là lệch config giữa train và render trong cùng pipeline. Rủi ro đó tái xuất hiện nếu bạn train bằng gsplat rồi render bằng đường Inria cũ (mix codebase). Cách né đúng: train và render test poses đều trong gsplat, cùng cờ — khi đó EWA filter của gsplat có khác Inria từng bit cũng không sao, vì tính nhất quán nội bộ mới là thứ quyết định. Sanity check bắt buộc trước khi tin: render lại vài view train bằng đúng đường render-submission, so PSNR với số eval lúc train; lệch >0.1dB là có mismatch. Cái này bạn tự kiểm được trong 1 buổi, không cần hỏi ai.
4. Đúng, tôi chưa tính chi phí này — và nó thật. gsplat lưu checkpoint .pt, không phải .ply kiểu GaussianModel. Có utility export sang .ply trong các bản gsplat gần đây, nhưng theo logic câu 3, export sang .ply để render bằng pipeline Inria là chính cái mismatch cần tránh. Nghĩa là phải viết lại 04_render_test_poses.py trên rasterization API của gsplat (đọc test poses COLMAP → render → lưu ảnh đúng tên/format submission) và giữ 06_package_submission.py gần như nguyên. Ước lượng thực tế: 1–2 ngày cho người quen code + nửa ngày sanity check. Phase 1 sửa lại thành ngày 3–11, và đây là việc làm trước khi đốt GPU-giờ vào sweep, không phải sau.
5. Thẳng thắn: đó là đoán có cấu trúc, không có benchmark thật chống lưng. Không tồn tại số liệu công khai nào cho scene BTS private của BTC. Ba thứ có thật đằng sau con số: bảng độ nhạy (toán chính xác từ công thức chấm), delta công bố của từng kỹ thuật trên benchmark chuẩn (MipNeRF360...), và Top1=74.9 chứng minh mức đó đạt được. Nhưng bước "chuyển delta từ MipNeRF360 sang drone-BTS" là giả định thuần túy. Cách dùng đúng con số 70–74: coi nó là prior để bị Phase 0 cập nhật, không phải dự báo. Sau khi có trung bình 5 scene public + 1 lần nộp private calibrate, vứt con số của tôi đi và dùng số của bạn.
6. Bạn đúng một nửa — và cái sai là do tôi viết không rõ. Diagnostic tương quan chỉ là đọc lại metric per-image + tính khoảng cách camera, chi phí gần bằng 0, nên chạy trên cả 5 scene public, không lý do gì gate ở 2. Con số "2 scene" trong bài trước là dev set cho các A/B training run tốn GPU-giờ — hai chuyện khác nhau bị tôi viết dính vào nhau. Gate sửa lại: tương quan âm nhất quán trên đa số 5 scene, cộng kiểm tra mắt thường 5 ảnh tệ nhất mỗi scene xem có đúng là floaters vùng khuyết không (Pearson trên khoảng cách Euclid là proxy yếu — pose xa nhưng cùng hướng nhìn vẫn dễ, pose gần nhưng góc lạ mới khó; nếu muốn kỹ hơn thì đo góc lệch hướng nhìn tối thiểu thay vì khoảng cách).
Trả lời câu cuối của bạn: không cần gửi đi đâu cả. Câu 1, 5, 6 đã được trả lời và nhận sai ở trên. Câu 2, 3, 4 thuộc loại kiểm bằng tay rẻ hơn hỏi bất kỳ AI nào — một buổi đọc source gsplat (export_splats, rasterization API, cờ --antialiased) cộng một run khói 1M Gaussian là ra cả ba đáp án bằng số thật. Đúng tinh thần bạn đã áp cho nhánh mip-splatting.