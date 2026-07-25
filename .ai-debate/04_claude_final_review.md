# 04 Claude Final Review

## 0. Phạm vi và phương pháp verify

Tôi đã đọc theo đúng thứ tự yêu cầu: `trao đổi.md` → `01_codex_proposal.md` → `02_claude_review.md` → `03_codex_response.md`, sau đó tự trích xuất cell-level evidence từ các file `.ipynb` bằng `python3 -c "import json; ..."` (không dùng `cat`/`sed` trên JSON thô). Trọng tâm verify độc lập là dòng gây tranh cãi nhất trong bảng Adjudication của `03_codex_response.md`:

> *"`B2_done 2.ipynb` và `B2_done 2 safe.ipynb` có cell override `LOW_VRAM_PROFILE=0`, nhưng kết quả cuối vẫn là `0.4027`"* → verdict `PARTIAL`.

Đây là claim tôi bác bỏ có bằng chứng cụ thể ở mục 1 dưới đây, và nó là bản lề của toàn bộ phần còn lại của review này.

---

## 1. Ba điểm yếu lớn nhất của Experiment Ladder (`03_codex_response.md`)

### 1.1. Ladder 2 (B2 Failure Isolation) dựa trên một tiền đề đã sai — Codex tự downgrade một hypothesis mà chính bằng chứng của Codex không hỗ trợ

**Verify trực tiếp bằng `python3 -c "import json; ..."` trên hai file:**

- `downloads/B2_done 2.ipynb`
- `downloads/B2_done 2 safe.ipynb`

Kết quả (0-based cell index trong JSON `cells[]`):

```
downloads/B2_done 2.ipynb
  cell idx=13  execution_count=None  outputs=0   # "CELL X - Configure depth supervision / train profile"
                                                  # DISABLE_LOW_VRAM_PROFILE = True → extra_train_env['LOW_VRAM_PROFILE']='0'

downloads/B2_done 2 safe.ipynb
  cell idx=14  execution_count=None  outputs=0   # cùng cell, DISABLE_LOW_VRAM_PROFILE=False, SAFE_DENSIFY_OVERRIDE=True
                                                  # → extra_train_env['LOW_VRAM_PROFILE']='1' + densify schedule tường minh
```

Cả hai cell override đều có `execution_count = None` và `outputs = []` — nghĩa là **chưa từng được chạy** trước khi notebook được lưu ở trạng thái hiện tại. Đây không phải suy luận, mà là thuộc tính `.execution_count`/`.outputs` đọc thẳng từ JSON của chính file mà Codex trích dẫn làm bằng chứng.

Quan trọng hơn: biến `extra_train_env` — biến duy nhất mang giá trị override vào lệnh train thật (`env.update(extra_train_env)`) — **chỉ được định nghĩa trong đúng cell chưa-từng-chạy đó**. Tôi grep toàn bộ notebook cho `extra_train_env` và không tìm thấy bất kỳ định nghĩa nào khác, không có `os.environ['LOW_VRAM_PROFILE']` trực tiếp, không có `%env`, và `pipeline/scripts/05_run_b2_pilot.sh` không tự source `default.env` hay set biến này. Cell train thật (idx=22 trong `B2_done 2.ipynb`, in ra `Score mean: 0.4027`) chạy được (`execution_count=18`, có output) chỉ vì `extra_train_env` đã tồn tại trong bộ nhớ kernel từ một **phiên bản trước đó (đã bị ghi đè) của cell 13** — hành vi kinh điển của Jupyter: sửa source một cell sau khi đã chạy sẽ xoá `execution_count`/`outputs` của riêng cell đó, nhưng không xoá kết quả của các cell downstream đã chạy trước đó với giá trị biến cũ.

Hệ quả: **không thể biết** `LOW_VRAM_PROFILE` thực sự bằng gì khi Score 0.4027 được tạo ra trong hai file này (phần log echo `LOW_VRAM_PROFILE=...` ở đầu `03_train_3dgs.sh` — dòng 214-216 — nằm trong đoạn output đã bị cắt bởi chính wrapper: `"Kết quả truyền trực tuyến bị cắt bớt đến 5000 dòng cuối"`, xác nhận bằng output thật của cell idx=22). Điều duy nhất chắc chắn: **giá trị override `LOW_VRAM_PROFILE=0` mà Codex dùng làm bằng chứng KHÔNG được áp dụng** cho run tạo ra 0.4027, vì cell tạo ra nó chưa từng thực thi.

Để đối chứng, tôi tìm được bằng chứng ngược lại đủ mạnh ở file gốc `B2_done.ipynb` **tại root repo** (không phải bản trong `downloads/`, hai file này khác nhau — `md5sum` khác, kích thước khác: root 1,282,217 bytes vs `downloads/B2_done.ipynb` 1,294,207 bytes). Trong `B2_done.ipynb` (root), output thật (đã chạy, có `execution_count`) in nguyên văn:

```
[b2-train-render-eval] LOW_VRAM_PROFILE=1
[b2-train-render-eval] RESOLUTION=4
[b2-train-render-eval] DENSIFY_UNTIL_ITER=0
```

Đây chính xác là dòng Claude (`02_claude_review.md` mục 1.6) trích dẫn — xác nhận đúng, chỉ khác đường dẫn file cụ thể (root, không phải `downloads/`). Vậy: **có ít nhất một run failed (`~0.40`) được xác nhận chắc chắn 100% chạy với `LOW_VRAM_PROFILE=1`/densify tắt**, và **không có run failed nào trong toàn bộ 4 notebook `B2_done*` đã thực sự chạy với `LOW_VRAM_PROFILE=0` được xác nhận**. Điều Codex mô tả là "đã thử override và vẫn fail" — sự kiện then chốt khiến Codex hạ hypothesis của Claude từ có-thể-đúng xuống `PARTIAL` — **chưa từng xảy ra trong dữ liệu hiện có**.

**Tại sao đây là điểm yếu số 1 của Ladder:** Ladder 2 được thiết kế đúng đắn ("chạy `prepared` control, ép tường minh `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`..."), nhưng phần lý do Codex dùng để giữ nó ở mức "PARTIAL" (thay vì "gần như chắc chắn là thủ phạm chính") là sai lệch do đọc nhầm một cell chưa chạy thành một cell đã chạy. Sửa lại: confound `LOW_VRAM_PROFILE` hiện **chưa hề bị bác bỏ bởi bất kỳ run thật nào** — nó vẫn là giả thuyết số 1, mạnh hơn cách Codex trình bày, và Ladder 2 phải được ưu tiên tuyệt đối ngay sau Ladder 0/1, không cần "chờ" thêm bằng chứng nào nữa để tin nó đáng làm trước tiên.

### 1.2. Ngưỡng GO/STOP định lượng được đặt trước khi biết nhiễu đo lường — thứ tự Ladder tự mâu thuẫn logic

`03_codex_response.md`, Ladder 1: *"GO nếu score nằm trong `±0.005` quanh `0.6731`"*. Ladder 2: *"GO nếu score quay lại gần baseline"* (không định lượng). Ladder 3 (**chạy SAU** Ladder 1): *"nếu seed std `>0.005`, mọi so sánh một-run bị hạ độ tin cậy"*.

Vấn đề: ngưỡng `±0.005` ở Ladder 1 được dùng để **quyết định GO/STOP cho việc rebuild baseline** trước khi Ladder 3 đo được seed-to-seed std thật. Nếu std thật hoá ra, ví dụ, `0.015` (hoàn toàn có thể với 3DGS train trên 50 test images, densify stochastic, exposure init ngẫu nhiên), thì ngưỡng `±0.005` ở Ladder 1 là quá chặt một cách tuỳ tiện — reproduce baseline có thể "STOP" nhầm dù chênh lệch chỉ là nhiễu tự nhiên. Đây là lỗi trình tự nhân quả: **bậc đo nhiễu (Ladder 3) phải đứng trước bậc dùng ngưỡng phụ thuộc vào nhiễu (Ladder 1's `±0.005` gate)**, hoặc ít nhất Ladder 1 phải ghi rõ ngưỡng này là tạm thời và sẽ hiệu chỉnh lại sau Ladder 3. `01_codex_proposal.md` mục 4.3 cũng đặt cùng vấn đề (`std > 0.005` là ngưỡng để tin/không tin single-run) nhưng không đối chiếu ngược lại các ngưỡng `±0.005`/`+0.01` đã dùng trước đó ở mục 4.1/4.2. Không văn bản nào trong 3 bản giải thích **tại sao** `0.005` hay `0.01` được chọn (không có căn cứ thống kê, không có công thức propagation-of-error từ n=50 test images) — đây là **estimate thiếu căn cứ** lặp lại xuyên suốt cả 3 văn bản, không riêng gì `03`.

### 1.3. Ladder 4 (Oracle Ceiling) dùng một ngưỡng nhị phân cứng (`>=0.80` / `<0.75`) cho một phương pháp oracle chưa được đặc tả đủ để tin số ra là "trần thật"

Ladder 4 nói: *"GO nếu oracle `>=0.80`... STOP nếu oracle `<0.75`"*, nhưng "Việc làm" chỉ ghi ngắn gọn "geometry-assisted warp/blend oracle". Không văn bản nào (kể cả `01`, kể cả review phản biện của Claude ở `02` mục 5.B) đặc tả đủ chi tiết: oracle dùng bao nhiêu neighbor views, có occlusion-aware hay không, có Poisson-blend hay chỉ nearest-splat, có làm mù render bằng exposure calib không. Một oracle 1-view naive-warp (không occlusion-aware) sẽ **đánh giá thấp trần thật** một cách hệ thống (lỗ hổng occlusion tạo artifact giả), trong khi một oracle multi-view blend tốt có thể vênh nhau `0.05-0.10` điểm Score chỉ do implementation choice — đủ để đổi hẳn quyết định GO/STOP ở ranh giới `0.75-0.80`. Đặt một gate quyết định-toàn-bộ-roadmap (tiếp tục 3DGS hay nhảy sang representation/ensemble) lên một con số chưa xác định rõ cách đo là rủi ro thực sự: **kết quả của Ladder 4 phụ thuộc mạnh vào lựa chọn thực thi của người chạy oracle, nhưng ngưỡng quyết định lại được viết như một hằng số khách quan**. Ladder 4 cần kèm một sub-spec bắt buộc (số neighbor views tối thiểu, phương pháp blend, và một "oracle sanity check" — ví dụ so oracle trên chính 200 train views với ground truth train — trước khi tin số ra trên 50 test views).

---

## 2. Thí nghiệm thiếu đối chứng / estimate thiếu căn cứ trong toàn chuỗi 3 văn bản

| # | Vấn đề | Bằng chứng | Vì sao thiếu control |
|---|---|---|---|
| 1 | **Checkpoint sweep chọn "best" bằng cách lặp lại eval trên đúng 50 ảnh test GT nhiều lần** | `trick/scripts/03_checkpoint_sweep.sh` dòng 29-45: với mỗi `ITER_DIR` (thường 8 mốc: `2000,4000,...,30000` theo `03_train_3dgs.sh` khi `LOW_VRAM_PROFILE=1`), script gọi `render_round1_test_poses.py` rồi `eval_round1_metrics.py --dataset_root .../public_set` — **trực tiếp trên `test/images` GT thật** — rồi chọn checkpoint có score cao nhất. `01_codex_proposal.md` mục 4.2 và `03_codex_response.md` Ladder 3 đều coi đây là bước "Tier 1, rẻ, an toàn" | Đây là **multiple-comparison / winner's-curse trên chính test set n=50**: chọn max trong 8 lần (rồi nhân thêm với 3 seed ở Ladder 3 = 24 lần) đánh giá trên cùng 50 ảnh sẽ cho một con số "best" lạc quan hơn khả năng tổng quát hoá thật, ngay cả trên chính `round1 public`, chưa nói tới private/round2. Không văn bản nào đề xuất correction (ví dụ: xác nhận checkpoint tốt nhất bằng một inference run độc lập / khác seed, thay vì tự tin luôn vào con số đã dùng để chọn nó) |
| 2 | **Oracle ceiling (Ladder 4) không có oracle-of-oracle để tự kiểm chứng phương pháp warp/blend đúng** | `03_codex_response.md` Ladder 4 "Việc làm" chỉ có 1 dòng, không có bước sanity-check trên train set | Nếu implementation oracle có bug (occlusion, chuẩn hoá màu sai), kết quả `<0.75` có thể là bug của oracle, không phải trần thật của dữ liệu — nhưng ladder dùng thẳng số đó làm gate nhị phân cho toàn bộ roadmap |
| 3 | **`diagnose_distance.csv` (dùng làm bằng chứng phản bác giả thuyết nền của B2 trong `02_claude_review.md` mục 1.4, được Codex accept `PARTIAL`)** | Script sinh ra nó (`09_diagnose_distance.py`) đã bị xoá ở commit `eb17653`; `git log --all -- "*09_diagnose_distance*"` không còn cách audit lại công thức "khoảng cách tới camera train gần nhất" được tính thế nào | Không ai trong cả 3 văn bản đề xuất **viết lại script để tái tạo con số này**, dù nó được dùng làm bằng chứng (dù yếu) để hạ ưu tiên giả thuyết nền tảng của toàn bộ nhánh `B2` |
| 4 | **Ladder 3 "3 seeds" không kiểm soát nguồn nhiễu nào khác ngoài seed** | Ladder 3 "Việc làm: checkpoint sweep; 3 seeds trên baseline control" — không nói rõ 3 seed này có cùng random init densify, cùng exposure init, cùng thứ tự shuffle batch hay không | Nếu "seed" chỉmeans thay `--seed`của 3DGS nhưng vô tình giữ nguyên các yếu tố phi-deterministic khác (CUDA non-determinism, order load ảnh trên các lần chạy khác nhau của Colab/Kaggle), std đo được có thể lẫn cả biến thiên môi trường thực thi, không thuần seed — không văn bản nào tách bạch hai nguồn này |
| 5 | **`PSNR_MAX` inconsistency (mục đã được cả 3 văn bản đồng thuận là bug thật)** | `downloads/B2_done*.ipynb` cell "reeval_latest" (idx=26 trong `B2_done 2.ipynb`) đặt `PSNR_MAX = 30.0`, cho `Score mean: 0.4449` — **cùng dữ liệu PSNR=10.5509** nhưng khác `psnr_max` — so với `Score mean: 0.4027` ở cell chính (`psnr_max=50.0` mặc định của `pipeline/scripts/eval_round1_metrics.py` dòng 32) | Tôi verify lại: đúng, cả hai số đều tồn tại thật trong cùng notebook (`B2_done 2.ipynb` cell idx=22/23 → `0.4027`; cell idx=26 → `0.4449`). Ladder 0 ("Measurement Lock") xử lý đúng bug này, nhưng KHÔNG audit ngược lại xem **baseline `0.6731`** đã từng bị tính bằng `psnr_max` nào khác `50.0` ở bất kỳ artifact cũ nào chưa — không có bước "grep toàn repo cho mọi giá trị `psnr_max` từng dùng" trong Ladder 0 |

---

## 3. Kiểm tra tiêu chí chốt: full-image score có thực sự là tiêu chí quyết định, hay crop/tower/skyline lén chen vào?

Về nguyên tắc văn bản, cả 3 bản đều tuyên bố đúng: `trao đổi.md` dòng 118 ("`full-image` là tiêu chí chốt"), `trick/README.md` dòng 33 ("metric tự động ưu tiên `full-image` vì đó là chỉ số thắng/thua cuối cùng"), `03_codex_response.md` Ladder 1/2/4/6A đều đo `full-image score` làm chỉ số GO/STOP chính.

Tuy nhiên có 3 chỗ mà crop-metric có nguy cơ **âm thầm trở thành tiêu chí quyết định thực tế**, dù không được văn bản gọi tên là "tiêu chí chốt":

1. **Ladder 5 (Region Diagnosis)**: GO/STOP dựa hoàn toàn trên `masked tower score` (không có full-image trong Ladder này) để quyết định có "mở local/tower branch" hay không. Điều này về nguyên tắc chấp nhận được (nó là bước *chẩn đoán*, không phải bước *chấm điểm cuối*), nhưng Ladder 6A/7 sau đó dùng chính quyết định "mở nhánh" này để đầu tư `2-4` hoặc `3-6` GPU-run — nếu tower score bị nhiễu (n nhỏ, mask bootstrap không chính xác — `trick/scripts/bootstrap_tower_masks.sh` tự sinh mask, không phải mask tay), toàn bộ nhánh tower specialist (Ladder 7) có thể được mở dựa trên một proxy-metric chưa qua full-image validation.
2. **Ladder 6A/6B ngưỡng dừng**: *"nếu sau 3 run chất lượng không vượt best control ít nhất `+0.01`, dừng"* — "chất lượng" không được định nghĩa là full-image score tường minh ở đây (khác với Ladder 1/2/4 luôn ghi rõ). Đây là khoảng hở ngôn ngữ: nếu người thực thi vô tình dùng skyline-crop (vì nhánh 6B test `mip-splatting` nhắm đúng bệnh skyline) làm chỉ số "vượt +0.01", quyết định có thể trôi khỏi full-image mà không ai để ý, vì văn bản không cấm rõ ràng.
3. **`experiment_matrix.csv`** (`trick/hcm0031/experiment_matrix.csv`) chỉ có cột `full_image_score`, không có cột `tower_crop_score`/`skyline_crop_score` — tự nó ép kỷ luật đúng hướng, nhưng đây cũng là lý do khiến `tower-crop 0.7064 > full-image 0.6731` (baseline) không bao giờ bị nhầm là "điểm chính thức" trong sổ sách — điểm cộng cho hạ tầng hiện tại, không phải cho ladder văn bản.

**Kết luận mục 3:** không có bằng chứng ai đang cố tình dùng crop metric thay full-image làm tiêu chí chốt cuối cùng, nhưng Ladder 6A/6B có khoảng hở ngôn ngữ ("chất lượng", "thắng control") đủ rộng để một người thực thi vô ý trôi khỏi full-image khi diễn giải "vượt +0.01". Cần sửa: **mọi ngưỡng GO/STOP dùng số tuyệt đối phải ghi rõ tên metric = full-image score**, không để ngầm hiểu.

---

## 4. Kiểm tra tính hợp lệ của các trick (leakage / dùng GT test để chọn model-checkpoint-blend weight)

Task yêu cầu đặc biệt tránh: **dùng GT test để lựa chọn mô hình/checkpoint/blend weight**. Tôi rà lại `trick/`, `01`, `02`, `03` theo đúng lăng kính này (không phải "có coi test set là benchmark hay không" — điều đó được cả 3 văn bản chấp nhận, mà là "có LẶP LẠI việc peek vào GT test để CHỌN giữa nhiều ứng viên hay không"):

- **`trick/scripts/03_checkpoint_sweep.sh` — VI PHẠM RANH GIỚI, dù nhẹ.** Đã nêu ở mục 2, dòng 1 của bảng: script chọn checkpoint "tốt nhất" bằng cách chấm điểm trực tiếp trên `test/images` GT thật rồi lấy max. Đây **chính xác** là "chọn checkpoint theo GT test" mà mục 9 của `01_codex_proposal.md` ("Legal/Leakage Boundary") lẽ ra phải liệt kê cấm, nhưng lại không liệt kê — mục 9 chỉ cấm "tuning fit trực tiếp theo GT test image" và "blend/compositing chọn trọng số theo GT test", coi checkpoint selection là nằm ngoài phạm vi cấm. Đây là một lỗ hổng logic: nếu "chọn trọng số blend theo GT test" bị cấm, thì "chọn checkpoint trong số N checkpoint theo GT test" là **cùng một loại hành vi** (model/hyperparameter selection bằng test signal), chỉ khác không gian tìm kiếm (blend-weight continuous vs checkpoint discrete). Vì `round1 public` GT được các văn bản coi là "benchmark chính thức để đối chiếu" (không phải final hidden test), rủi ro không phải "gian lận thi đấu" mà là **báo cáo optimistic bias**: con số "best checkpoint" sẽ luôn hoặc bằng hoặc cao hơn true single-run score, nên khi so sánh `B2` vs baseline sau này, phải đảm bảo **cả hai bên đều được sweep-checkpoint như nhau** (apples-to-apples), nếu không baseline "chưa sweep" sẽ thua giả tạo so với ứng viên "đã sweep". Không văn bản nào cảnh báo rõ điều này.
- **`trick/scripts/04_bootstrap_tower_masks.sh` / `05_run_m0_mask_eval.sh` — HỢP LỆ.** Mask sinh từ `tower_bbox3d.json` (geometry train-time, `pipeline/scripts/estimate_object_bbox3d.py`), không đọc GT test image để định nghĩa vùng crop. Đúng như `01` mục 9 yêu cầu ("mask... sinh từ train-time assets").
- **Ensemble / blend đề xuất ở cả 3 văn bản (Codex mục 6.2, Claude mục 5.F) — điều kiện hợp lệ được nêu đúng** ("blending rule không dùng GT test", "trọng số nghịch đảo độ lệch giữa các model tính từ chính render, không chạm GT test") nhưng **chưa có cơ chế kiểm tra tuân thủ**: không văn bản nào đề xuất một "audit script" tự động kiểm tra bất kỳ trọng số blend hay compositing rule nào có vô tình import/đọc file `test_poses.csv`/GT test hay không trước khi chấp nhận một PR/notebook mới. Đây là rủi ro quy trình (process risk), không phải rủi ro đã xảy ra.
- **`antenna-focus`/`depth-prior` (evidence từ `HCM0421`) — không phải leakage, nhưng là ngoại suy cross-scene/cross-protocol đã được cả `02` và `03` đồng thuận hạ cấp đúng.** Tôi xác nhận lại: `git show 9383e23:WORKLOG.md` ghi rõ `MODE=holdout`, và `pipeline/common/scenes.py` (tại `eb17653^`, tôi không re-verify sâu commit này lần nữa vì `03_codex_response.md` đã trích đúng `0f85a05:pipeline/common/scenes.py`) xác nhận `HCM0421` là Round 2. Đồng ý với cả 2 bên: đây là evidence yếu cho `hcm0031`, KHÔNG phải leakage.
- **"score-oriented loss" (Claude mục 5.I — thêm LPIPS trực tiếp vào training loss vì công thức chấm điểm có trọng số LPIPS cao nhất)** — về mặt kỹ thuật đây **không phải leakage** (loss dùng LPIPS giữa render và **train** GT, không đụng test GT), nhưng cần nói rõ: đây là "teaching to the test formula" ở mức hợp lệ (tối ưu đúng cái được đo bằng dữ liệu train hợp lệ), khác hẳn "teaching to the test images". Không văn bản nào trong `01`/`03` nhắc tới trick này dù nó rẻ và hợp lệ — đáng được thêm vào ladder chính thức (xem mục 5).

**Kết luận mục 4:** phát hiện leakage cụ thể duy nhất là `checkpoint_sweep.sh` chọn "best" bằng cách lặp eval trên GT test — không phải gian lận nghiêm trọng (vì đây là public benchmark có GT theo đúng thiết kế round1, không phải hidden test), nhưng là nguồn optimistic-bias thật cần được xử lý bằng kỷ luật so sánh (so cùng-điều-kiện-sweep), điều mà không văn bản nào trong 3 bản đã nêu rõ.

---

## 5. Thứ tự thí nghiệm đề xuất cuối cùng (sửa lại Ladder của Codex)

Giữ khung 8 bậc của Codex vì cấu trúc hợp lý, nhưng sửa 4 chỗ: (a) đưa Ladder 3 lên trước phần dùng ngưỡng phụ thuộc nhiễu của Ladder 1, (b) tách rõ Ladder 2 thành 2 lần chạy control thay vì 1, (c) thêm bước audit-leakage cho checkpoint sweep, (d) thêm "score-oriented loss" thí nghiệm rẻ vào nhánh 6A.

```
L0. Measurement Lock (0 GPU)
    - Chuẩn hoá mọi báo cáo score về psnr_max=50.0.
    - Grep TOÀN REPO (không chỉ notebook mới) cho mọi giá trị psnr_max từng dùng,
      gắn nhãn rõ bất kỳ số nào không dùng 50.0.
    - Sửa nhãn "source_mode=raw" sai trong trick/hcm0031/experiment_matrix.csv
      thành "prepared-no-depth" (đã được cả 3 văn bản đồng ý cần sửa, chưa ai sửa).

L1-mini. Seed/Noise Probe RÚT GỌN (chạy song song L0, 1 GPU run ngắn, không cần full 30k iter)
    - Train baseline control 2 lần (2 seed khác nhau) đến ít nhất 15k iter để có ước lượng
      thô về std trước khi đặt ngưỡng ±0.005 cho L1 đầy đủ.
    - Mục đích: hiệu chỉnh ngưỡng GO/STOP của L1, không phải kết luận cuối về variance.

L1. Gold Baseline Rebuild (1 GPU run full)
    - Ngưỡng GO/STOP lấy từ L1-mini, không hard-code ±0.005 nếu std thô > 0.005.
    - Lưu đủ artifact: pipeline_train_flags.json, chkpnt30000.pth, iteration_30000/point_cloud.ply.

L2. B2 Failure Isolation — 2 RUN, KHÔNG PHẢI 1 (sửa quan trọng nhất so với Codex)
    - Run 2a: prepared source, LOW_VRAM_PROFILE=0 tường minh, RESOLUTION=-1, densify mặc định KHÔNG bị auto tắt.
    - Run 2b (control âm, để xác nhận lại lần fail cũ có tái lập được không):
      prepared source, LOW_VRAM_PROFILE=1 (giữ nguyên cấu hình auto-crippled), mọi thứ khác giống 2a.
    - Lý do bắt buộc phải có 2b: nếu chỉ chạy 2a và nó thắng, vẫn không chắc chắn 100%
      rằng chính LOW_VRAM_PROFILE là biến quyết định (có thể do trùng hợp với thay đổi khác
      giữa lần fail cũ và lần chạy mới, ví dụ phiên bản COLMAP, driver CUDA...).
      2b tái lập lại đúng điều kiện fail cũ trên CÙNG hạ tầng hiện tại để cô lập biến.
    - GO: 2a phục hồi gần baseline VÀ 2b tái lập được vùng ~0.40 → xác nhận nhân quả rõ ràng.
    - STOP: nếu cả 2a và 2b đều rơi vào vùng thấp như nhau → LOW_VRAM_PROFILE bị loại,
      lỗi nằm ở chỗ khác (nguồn colmap/dense, prepared source tự nó có vấn đề).

L3. Checkpoint + Seed Hygiene ĐẦY ĐỦ
    - Checkpoint sweep + audit leakage: BẮT BUỘC báo cáo song song 2 số cho baseline VÀ
      mọi ứng viên B2/representation mới: (a) score của checkpoint cuối (không sweep),
      (b) score của best-checkpoint-sau-sweep. Không được so "best sweep của ứng viên"
      với "checkpoint cuối không sweep của baseline".
    - 3 seed đầy đủ trên baseline control.

L4. Oracle Ceiling — kèm sub-spec bắt buộc
    - Tối thiểu 3-5 neighbor views, occlusion-aware (z-buffer/depth test), không phải nearest-1-view.
    - Sanity check: chạy cùng oracle trên 20-30 TRAIN views (ẩn khỏi chính oracle như held-out
      nội bộ) trước khi tin số đo trên 50 test views — nếu oracle-on-train cũng thấp bất thường,
      nghi ngờ implementation bug trước khi kết luận về trần dữ liệu.

L5. Region Diagnosis (masked tower eval, dùng để CHẨN ĐOÁN, không dùng làm GO/STOP cuối cùng độc lập)

L6A / L6B. 3DGS-family continuation HOẶC representation branch
    - Thêm vào L6A: "score-oriented loss" (LPIPS-term nhỏ vào training loss, dùng train GT,
      hợp lệ, rẻ, chưa ai thử) như một trong 2-4 run được phép.

L7. Two-Stage / Ensemble — giữ nguyên như Codex, chỉ audit thêm: mọi trọng số blend phải có
    log rõ ràng cho thấy nó KHÔNG được tính từ bất kỳ file nào chứa "test" trong path.
```

---

## 6. Tiêu chí định lượng, rõ ràng để chuyển từ B2/3DGS sang E/G (representation) hoặc ensemble/2-stage

Không chấp nhận ngôn ngữ "nếu không tốt thì chuyển hướng". Đề xuất bộ gate cứng sau, thay thế/làm rõ Ladder 4 và Ladder 6A/6B của Codex:

**Điều kiện BẮT BUỘC CHUYỂN sang representation mới (Ladder 6B) hoặc mở sớm 2-stage/ensemble (Ladder 7), bỏ qua tiếp tục refine 3DGS thuần — kích hoạt nếu THOẢ MÃN BẤT KỲ điều kiện nào sau:**

1. **Oracle ceiling (L4, đã qua sanity check train-set) `< 0.75` full-image score.**
2. **L2 hoàn tất (cả 2a và 2b) mà cả hai đều `<= 0.55` full-image score** — tức là ngay cả sau khi cô lập `LOW_VRAM_PROFILE`, nhánh `prepared/dense` vẫn không phục hồi về gần baseline `0.6731` → bản thân ý tưởng `prepared+depth`, không phải config bug, là vấn đề.
3. **Sau đúng 4 run trong L6A (không phải "3 run" mơ hồ của Codex — cố định thành 4: 1 prepared+depths, 1 LPIPS-loss, 1 exposure/pose refine, 1 checkpoint/seed tốt nhất tổng hợp), không có run nào đạt full-image score `>= best_control + 0.015`** (nâng từ `+0.01` của Codex lên `+0.015` để vượt hẳn biên nhiễu đo được ở L1-mini/L3 — nếu L3 cho thấy seed std `s`, ngưỡng thực tế phải là `max(0.015, 3*s)` để đảm bảo gain không phải nhiễu).

**Điều kiện được phép TIẾP TỤC 3DGS-family (không bị buộc chuyển hướng):**

- Oracle `>= 0.80` **VÀ** L2 cho thấy ít nhất một trong hai run (2a hoặc 2b) phục hồi về vùng `>= 0.60` **VÀ** trong 4 run của L6A có ít nhất 1 run vượt `best_control + max(0.015, 3*s)`.

**Vùng xám `0.75 <= oracle < 0.80`:** cho phép 1 vòng L6A rút gọn (tối đa 2 run, không phải 4) trước khi bắt buộc quyết định lại — không được dùng vùng xám này để trì hoãn vô thời hạn (đây chính là kiểu "ngôn ngữ mơ hồ" mà Codex mắc phải khi để khoảng `0.75-0.80` không có hành động rõ ràng, chỉ ngầm hiểu qua 2 ngưỡng GO/STOP riêng biệt mà không nói tới vùng giữa).

Các ngưỡng số cụ thể (`0.75`, `0.015`, `4 run`, `3*s`) là **đề xuất của tôi để thay thế ngôn ngữ định tính**, không phải con số đã được đo — nhưng khác với Codex, tôi neo chúng vào biến đo được thật (`s` = seed std từ L3) thay vì hằng số cứng không có cơ sở, đúng tinh thần "không chấp nhận ngôn ngữ mơ hồ" mà nhiệm vụ yêu cầu.

---

## Tổng kết verdict

Chiến lược tổng hợp của `03_codex_response.md` có khung tốt (Ladder 0→7, phân biệt fact/inference đúng phần lớn, oracle làm gate là ý đúng), nhưng có một lỗi verify sự thật nghiêm trọng làm lệch mức độ ưu tiên của Ladder 2: bảng Adjudication kết luận `PARTIAL` cho giả thuyết `LOW_VRAM_PROFILE` dựa trên việc đọc nhầm hai cell **chưa từng được thực thi** (`execution_count=None`, `outputs=[]`, xác minh trực tiếp trong `downloads/B2_done 2.ipynb` cell idx=13 và `downloads/B2_done 2 safe.ipynb` cell idx=14) thành bằng chứng đã-chạy-và-vẫn-fail. Ngoài ra Ladder có các lỗ hổng thật: ngưỡng GO/STOP đặt trước khi đo nhiễu (mục 1.2), oracle thiếu sub-spec (mục 1.3), checkpoint sweep dùng GT test lặp lại mà không audit optimistic bias (mục 2, mục 4), và ngôn ngữ GO/STOP ở Ladder 6A/6B có khoảng hở khiến crop-metric có thể lén thay thế full-image (mục 3). Không có lỗi nào trong số này vô phương cứu chữa — tất cả đều sửa được bằng cách điều chỉnh trình tự và thêm control, như đề xuất ở mục 5-6, không cần viết lại chiến lược từ đầu.

VERDICT: ACCEPT WITH CHANGES

Lý do: khung 8 bậc và nguyên tắc "oracle làm gate" của Codex là đúng hướng và nên giữ, nhưng bảng Adjudication chứa một lỗi verify bằng chứng cụ thể (đọc nhầm cell chưa chạy) làm suy yếu độ ưu tiên đúng đắn của Ladder 2, và ba lỗ hổng thiếu-control/ngưỡng-thiếu-căn-cứ (mục 1.2, 1.3, mục 2 dòng 1) phải được vá trước khi tiêu tốn GPU-hour theo đúng thứ tự đã sửa ở mục 5, nếu không rủi ro lặp lại chính xác kiểu sai lầm đo lường mà cả 3 vòng debate trước đã cùng nhau phát hiện và sửa cho nhau.
