# 07 Claude Audit

## Phạm vi và phương pháp

Đã đọc toàn bộ `trao đổi.md` (đặc biệt phần mới "## Tổng hợp tranh luận 2026-07-24", dòng 180-431) và `.ai-debate/06_final_diff.patch`, đối chiếu ngược với `.ai-debate/01` đến `05`, và verify trực tiếp các con số bằng cách tính lại trung bình từ `pipeline/work/hcm0031/eval_metrics_m0*.csv` (khớp chính xác `PSNR 21.6938 / SSIM 0.6819 / LPIPS 0.1542 / Score 0.6731` cho full-image, `0.7064` cho tower-crop, `0.6384` cho skyline-crop), kiểm tra tồn tại của `pipeline_train_flags.json` trong `pipeline/scripts/03_train_3dgs.sh` dòng 262, kiểm tra `trick/scripts/03_checkpoint_sweep.sh`, và kiểm tra 4 branch git (`feature/mip-splatting`, `feature/gsplat-mcmc`, `compact/compact-gaussian`, `feature/depth-anything-v2`) đều tồn tại thật trên `origin`.

## 1. Sai số liệu hoặc mâu thuẫn với bằng chứng

Không phát hiện lỗi số liệu quan trọng. Toàn bộ các con số cốt lõi trong phần mới của `trao đổi.md` — `Score 0.6731/0.7064/0.6384`, `Score 0.4027`, `depth_map_files=400`, `fused.ply` `121495051` bytes — khớp chính xác với `.ai-debate/01-05` và với dữ liệu CSV thật trong repo.

Có một điểm đáng ghi nhận (không phải "sai" nhưng là bỏ sót âm thầm khi tổng hợp từ `05_codex_synthesis.md` sang `trao đổi.md`): bản `05` (mục 4, mục 5.4) còn giữ dòng bằng chứng "sparse gốc từng có `388` ảnh... còn thiếu `138` frame không nằm ở train hay test cục bộ" (đây chính là phát hiện `02_claude_review.md` mục 1.3 gọi là "đòn bẩy tiềm năng lớn nhất"), và có hẳn một dòng ma trận "Data recovery từ sparse gốc | Có thể rất cao...". Bản cuối trong `trao đổi.md` chỉ còn sót lại cụm ngắn "xin hoặc phục hồi dữ liệu nếu có thể mở rộng từ sparse gốc `388` frame" ở mục 3.5, không còn con số `138/35.6%` và không còn dòng riêng trong ma trận mục 4. Đây không phải mâu thuẫn số liệu, nhưng là một context quan trọng bị làm loãng khi rút gọn từ `05` — nên khôi phục lại ít nhất con số `138/388` vì nó liên quan trực tiếp tới trần khả thi của `0.85` (xem thêm mục 6 audit này).

## 2. Đề xuất thiếu khả thi

Các branch được nhắc tới (`feature/mip-splatting`, `feature/gsplat-mcmc`, `compact/compact-gaussian`, `feature/depth-anything-v2`) đều tồn tại thật trong repo, và các script/artifact được tham chiếu (`pipeline_train_flags.json`, `trick/scripts/03_checkpoint_sweep.sh`, `04_bootstrap_tower_masks.sh`) đều có thật — không có phụ thuộc vào thứ không tồn tại.

Vấn đề đáng chú ý là ngân sách thời gian ở mục "### 6. Kế hoạch 24 giờ, 72 giờ và full experiment" (`trao đổi.md` dòng 326-359). Bucket "24 giờ" gộp chung: (1) Measurement Lock, (2) Rebuild `gold baseline` (một lần train 30k iteration đầy đủ), (3) "Chạy `B2 failure isolation`", (4) xác nhận artifact — mà không kèm bất kỳ con số GPU-hour/số phiên nào. Điều này bỏ qua một ràng buộc hạ tầng đã được chính `02_claude_review.md` dòng 226 nêu rõ trước đó: *"ngân sách GPU thực tế là Kaggle/Colab free-tier (phiên ~9-12h, T4/P100, VRAM hạn chế...)"*. Nếu áp đúng sửa đổi quan trọng nhất của `04` (xem mục 3 audit này) — `B2 failure isolation` phải là **2 run** đầy đủ (bao gồm khả năng phải dựng lại `dense stereo` cho ít nhất một run) chứ không phải 1 — thì việc nhét "rebuild gold baseline + 2 run B2 isolation + audit artifact" vào đúng một bucket "24 giờ" trên hạ tầng free-tier có session cap là một ước tính lạc quan chưa được kiểm chứng bằng con số cụ thể nào. Đây không phải lỗi nghiêm trọng (không hard-code một cam kết sai), nhưng là một khoảng hở feasibility nên được vá bằng cách ghi rõ số GPU-hour ước tính hoặc nới bucket "24 giờ" thành khoảng linh hoạt hơn.

## 3. Thiếu thí nghiệm đối chứng (control) — trọng tâm: có giữ được sửa đổi "2 run" của `04` không?

Đây là phát hiện quan trọng nhất của audit này: **sửa đổi quan trọng nhất của `04_claude_final_review.md` (L2 phải là "2 RUN, KHÔNG PHẢI 1") chỉ được giữ lại một phần, và bị pha loãng ngay tại các điểm vận hành then chốt của bản tổng hợp cuối.**

`04` viết rất rõ (mục 5, `L2`):

> *"Run 2a: prepared source, LOW_VRAM_PROFILE=0 tường minh... Run 2b (control âm, để xác nhận lại lần fail cũ có tái lập được không): prepared source, LOW_VRAM_PROFILE=1... Lý do bắt buộc phải có 2b: nếu chỉ chạy 2a và nó thắng, vẫn không chắc chắn 100% rằng chính LOW_VRAM_PROFILE là biến quyết định... GO: 2a phục hồi gần baseline VÀ 2b tái lập được vùng ~0.40 → xác nhận nhân quả rõ ràng."*

Trong `trao đổi.md` phần mới, ý tưởng 2-run chỉ còn xuất hiện dưới dạng một bullet mô tả (mục 3.1, dòng 239-241):

> *"cách ly failed run `B2` bằng control rõ ràng: `prepared`, `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1` / so với control âm tái hiện cấu hình low-VRAM cũ"*

Nhưng khi đi tới các phần **vận hành thật** — nơi quyết định thực sự được đưa ra — thiết kế 2-run biến mất:

- Mục "24 giờ" (dòng 333): *"3. Chạy `B2 failure isolation`"* — số ít, không nói rõ 2 run.
- Mục 7 "GO/STOP thresholds" (dòng 367-368): *"`GO B2 isolation`: run `prepared` control phục hồi về gần baseline"* — chỉ nhắc **một** run (tương đương 2a của `04`), hoàn toàn không yêu cầu control âm (2b) phải tái lập lại vùng `~0.40` để xác nhận quan hệ nhân quả. Đây đúng là kịch bản `04` cảnh báo: *"nếu chỉ chạy 2a và nó thắng, vẫn không chắc chắn 100%..."*.
- Mục 8 "Điều kiện chuyển sang E/G" (dòng 392): *"`B2 failure isolation` không cứu được branch `prepared`"* — không định nghĩa "không cứu được" nghĩa là gì trong khuôn khổ 2-run, để ngỏ khả năng người thực thi chỉ chạy 2a, thấy nó không tăng, rồi kết luận (sai) rằng `LOW_VRAM_PROFILE` bị loại — trong khi chưa hề chạy 2b để biết liệu vùng fail cũ có tái lập hay không.

**Kết luận mục 3:** bản tổng hợp cuối *biết* về yêu cầu 2-run (còn ghi trong danh mục phương án) nhưng *không đưa nó vào* ngôn ngữ GO/STOP và kế hoạch thực thi — đúng loại lỗi mà cả `04` và toàn bộ chuỗi debate đã cố sửa (đọc nhầm/rút gọn 1 run thành đủ bằng chứng nhân quả). Cần sửa: mục 7 "GO B2 isolation" phải viết lại thành "cả 2a phục hồi gần baseline VÀ 2b tái lập vùng ~0.40", và mục 6/mục 8 phải nói rõ "2 run" thay vì số ít.

## 4. Test leakage hoặc trick trái luật

Nội dung xử lý rủi ro `checkpoint sweep` (mục 9, dòng 406-421) về cơ bản đúng tinh thần sửa của `04`: liệt kê đúng rủi ro *"chọn checkpoint tốt nhất bằng cách lặp eval trên `test/images` GT thật"* và đưa ra kiểm soát cụ thể *"báo song song: checkpoint cuối; best checkpoint sau sweep"* + *"không so best-sweep ứng viên với checkpoint-cuối baseline"* — đây chính xác là biện pháp apples-to-apples mà `04` mục 1 dòng 1 và mục 5 (`L3`) yêu cầu. Không phải chỉ nhắc chung chung.

Có một điểm chưa nhất quán nội bộ đáng lưu ý: ma trận Expected Gain/Cost/Risk (mục 4, dòng 292) ghi hàng `Checkpoint sweep | Trung bình | Thấp | Thấp | ...` — cột **Risk** để "Thấp", trong khi `04` mục 4 gọi đúng hành vi này là *"VI PHẠM RANH GIỚI, dù nhẹ"* và là nguồn *"optimistic-bias thật"*. Cột Evidence của hàng này (*"repo đã có script sweep, chưa có run matrix thật"*) cũng không nhắc gì tới rủi ro leakage đã được xử lý riêng ở mục 9. Nếu người đọc chỉ lướt ma trận mục 4 mà bỏ qua mục 9, họ dễ hiểu nhầm `checkpoint sweep` không có rủi ro về tính hợp lệ đo lường. Nên thêm chú thích chéo (cross-reference) từ hàng ma trận sang mục 9, hoặc nâng Risk lên "Thấp-trung bình" kèm ghi chú "risk đo lường, không phải risk kỹ thuật".

## 5. Kế hoạch không tối ưu full-image score

Không phát hiện vấn đề quan trọng. Mục 7 (dòng 363) mở đầu bằng câu chốt rõ ràng: *"Mọi threshold trong phần này mặc định nói về `full-image score`, không dùng crop metric thay thế"* — đây chính xác là bản sửa `04` yêu cầu ở mục 3 (*"mọi ngưỡng GO/STOP dùng số tuyệt đối phải ghi rõ tên metric = full-image score"*), và nó bao trùm toàn bộ các GO/STOP liệt kê sau đó (không còn khoảng hở kiểu "chất lượng"/"thắng control" mơ hồ như bản `03` bị `04` phê bình). Mục 8 dùng `masked tower eval` như một trong các điều kiện *mở nhánh* E/G (chẩn đoán để quyết định mở thí nghiệm mới), không dùng nó thay `full-image score` để chấm điểm thắng/thua cuối cùng — đúng ranh giới `04` mục 3 chấp nhận là hợp lệ.

## 6. Nhận định quá chắc chắn về mục tiêu 0.85

Không phát hiện vấn đề quan trọng mới do bước tổng hợp này gây ra. Các câu chốt đều được hedge đúng mực: *"cần coi `0.85` là target, không phải kỳ vọng mặc định"* (dòng 214, 425) và *"chưa có bằng chứng nào đủ mạnh để tuyên bố chắc chắn một hướng cụ thể sẽ đạt `0.85`"* (dòng 426) — không có câu nào tuyên bố dứt khoát "chắc chắn đạt" hay "chắc chắn không đạt".

Câu duy nhất mang giọng tự tin cao là dòng 213: *"tuning nhẹ gần như chắc chắn không đủ để tự tin nhắm `0.85`"*. Câu này không phải phát sinh mới ở bước tổng hợp — nó được kế thừa gần như nguyên văn từ `03_codex_response.md` (dòng 18, 369: *"tuning nhẹ gần như chắc chắn không đủ cho `0.85`"*), đã qua ít nhất 2 vòng debate mà không bị `04` phản bác cụ thể, và dùng từ hedge ("gần như") chứ không tuyệt đối. Với biên độ gap thật (`0.6731` → `0.85` là +0.177, trong khi các delta tuning từng đo được trên `HCM0421` chỉ ở mức phần nghìn) đây là một suy luận có cơ sở hợp lý dù chưa có oracle riêng cho `hcm0031`. Không cần sửa, nhưng nên gắn thêm một câu ngắn nhắc rằng nhận định này dựa trên ngoại suy từ delta tuning ở `HCM0421`, không phải oracle đo trực tiếp trên `hcm0031`, để nhất quán với chính nguyên tắc "không ngoại suy cross-scene" mà văn bản này đang áp dụng cho các trường hợp khác (`depth-prior`/`antenna-focus`).

---

## Tổng kết

Bản tổng hợp cuối trong `trao đổi.md` giữ số liệu chính xác, xử lý tốt yêu cầu "full-image làm tiêu chí chốt" và "không hard-code ngưỡng trước khi đo nhiễu", và có cơ chế kiểm soát leakage checkpoint-sweep hợp lý ở mục 9. Tuy nhiên, phát hiện quan trọng nhất mà `04_claude_final_review.md` để lại — B2 failure isolation phải là **2 run** (positive override + control âm tái lập điều kiện fail cũ) mới đủ để khẳng định nhân quả `LOW_VRAM_PROFILE` — chỉ còn là một dòng mô tả trong danh mục phương án (mục 3.1), và bị rút gọn về một run duy nhất trong chính các đoạn quyết định vận hành: kế hoạch 24 giờ (mục 6), ngưỡng GO/STOP (mục 7), và điều kiện chuyển sang E/G (mục 8). Đây là đúng dạng lỗi mà toàn bộ chuỗi 3 AI đã dày công phát hiện và sửa cho nhau ở các vòng trước, nên việc nó bị pha loãng ở bước tổng hợp cuối là một hồi quy (regression) đáng phải sửa trước khi bắt đầu tiêu tốn GPU-hour thật.

KẾT LUẬN: NEEDS REVISION — cần sửa mục 6/7/8 của `trao đổi.md` để yêu cầu tường minh 2 run (2a override + 2b control âm) cho B2 failure isolation trước khi coi văn bản là sẵn sàng thực thi; các điểm còn lại (mục 1, 2, 4, 5, 6 của audit này) chỉ là các vá nhỏ, không chặn việc bắt đầu Measurement Lock/Gold Baseline Rebuild song song.
