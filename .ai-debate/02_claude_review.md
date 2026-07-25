# 02 Claude Review

## 0. Tóm tắt điều hành (verdict trước khi đọc Codex, giữ nguyên sau khi đọc)

Sau khi tự đọc `trao đổi.md`, toàn bộ `trick/`, `pipeline/scripts/`, các CSV trong `pipeline/work/hcm0031/`, và đối chiếu với `git log`/`git show` cho các commit được nhắc tới, chẩn đoán độc lập của tôi **không trùng khớp** với khung `P0 -> M0 -> B2 -> C/F -> A` mà repo đang neo vào. Ba phát hiện quan trọng nhất — mà cả `trao đổi.md` lẫn `01_codex_proposal.md` đều **không nêu** — là:

1. **`B2` (dense-stereo/prepared-source) đã được thử thật trong repo và thất bại thảm khốc, lặp lại 2 lần** (`PSNR` sập từ `21.69` xuống `~10.5 dB`, `Score ~0.40`), bằng chứng nằm ngoài `pipeline/work/hcm0031/` (ở `downloads/B2_done.ipynb`, `downloads/B2_done 2 safe.ipynb`, `colab_b2_results_drop/`) — cả tài liệu `trao đổi.md` (viết trước) và bản đề xuất của Codex (không quét các thư mục này) đều bỏ sót.
2. Có một **nghi phạm cấu hình cụ thể** cho thất bại đó: `LOW_VRAM_PROFILE=1` + `RESOLUTION=4` (ảnh còn 1/4 độ phân giải) và theo code, nhánh này **tắt hoàn toàn densification** (`DENSIFY_UNTIL_ITER=0`, `DENSIFY_FROM_ITER=0`, `DENSIFICATION_INTERVAL=0`, `PERCENT_DENSE=0.0`) nếu không override — trong khi baseline "tốt" (`0.6731`) lại không dính lỗi này. Đây là confound bắt buộc phải tách trước khi kết luận bất cứ điều gì về `B2`/depth.
3. **Bằng chứng dùng để hạ ưu tiên `depth-prior`/`antenna-focus`** (`8e88bc0`, `9383e23`) đến từ `HCM0421` — một scene **Round 2**, dùng **holdout tự tạo từ ảnh train** (không có GT thật), theo đúng docstring `pipeline/common/scenes.py`: *"Round 2 (BTC đã bỏ round 1, thi lại từ đầu)... KHÔNG scene nào có ground-truth thật"*. Áp kết luận đó cho `hcm0031` (Round 1 public, có GT thật) là một phép ngoại suy chưa được kiểm chứng, không phải "bằng chứng thật duy nhất" như `trao đổi.md` tự mô tả.

Ngoài ra, `Score = 0.85` là một target cần được đặt câu hỏi nghiêm túc về tính khả thi toán học/thực nghiệm (xem mục 3.9), và có một **giới hạn dữ liệu cấu trúc** chưa ai nhắc tới: 138/388 (35.6%) khung hình có pose COLMAP đã đăng ký nhưng không hề tồn tại ở bất kỳ đâu trong `public_set` cục bộ (không phải train, không phải test) — nghĩa là quỹ đạo camera train thật sự "rách" hơn những gì `full-image` gợi ý.

---

## 1. Chẩn đoán độc lập (trước khi đọc Codex)

### 1.1. Baseline `0.6731` không phải "raw baseline thuần"

`pipeline/work/hcm0031/gs_model/cfg_args`:

```
Namespace(sh_degree=3, source_path='/kaggle/working/pipeline/work/hcm0031/colmap/dense',
model_path='...', images='images', depths='', resolution=-1, ..., train_test_exp=False, ...)
```

Baseline đang được coi là "mốc" (`Score 0.6731`, `pipeline/work/hcm0031/eval_metrics.csv` = `eval_metrics_m0.csv`, khớp từng số) **đã train từ `colmap/dense`** (nguồn undistort qua `image_undistorter`), **không phải** `train/sparse/0` gốc thuần. `depths=''` → không có depth supervision, và log xác nhận:

```
$ grep -o "Depth Loss=[0-9.]*" pipeline/work/hcm0031/03_train_3dgs.log | sort -u
Depth Loss=0.0000000
```

`resolution=-1` (không ép resolution) và `cameras.json[0].width=1310` ≈ ảnh gốc `1320x989` → baseline train ở **gần full-res**, và `point_cloud/iteration_15000/point_cloud.ply` = **1.23 GB** so với `input.ply` = **3.97 MB** → densification **có chạy** (từ 147065 điểm khởi tạo lên hàng triệu Gaussian). Baseline này "sạch" về cấu hình huấn luyện cơ bản — nhưng tên gọi "raw baseline" trong `trick tham khảo.md`/`trick/README.md` (dòng "so `raw source` vs `prepared source`") là sai lệch: **baseline đang dùng đã là "prepared" rồi**, nên toàn bộ khung so sánh raw-vs-prepared trong `experiment_matrix.csv` (`baseline_ref,...,source_mode=raw,...,0.6731`) tự mâu thuẫn với chính `cfg_args` (`source_path=.../colmap/dense`). Cột `source_mode=raw` trong `experiment_matrix.csv` là **sai nhãn**, chưa ai sửa.

### 1.2. Artifact cục bộ không đủ để tự verify `0.6731`

`pipeline/work/hcm0031/gs_model/point_cloud/` chỉ có `iteration_15000` (không có `iteration_30000`, không có `chkpnt*.pth`), trong khi `pipeline/work/hcm0031/04_render_test_poses.log` dòng đầu ghi rõ model dùng để render ra `0.6731` là:

```
[21:58:14] Model: /kaggle/working/.../gs_model/point_cloud/iteration_30000/point_cloud.ply
```

→ checkpoint 30k thật (cái tạo ra số `0.6731`) **không tồn tại cục bộ**, không thể tự tay verify lại từ máy hiện tại, phải tin vào log/CSV đã lưu.

### 1.3. `P0 PASS` chỉ kiểm tra tính nhất quán cục bộ, không kiểm tra tính đầy đủ so với gốc

`trao đổi.md` dòng 67–78 nói `P0` cho `hcm0031` "đã PASS" với `train/images count=200, unique=200, suspicious=no`. Nhưng khi COLMAP mapper/undistort chạy trên sparse gốc, log thật lại nói khác:

```
pipeline/work/hcm0031/01_run_colmap.log:
"Dùng sparse có sẵn: .../hcm0031/train/sparse/0 (388 ảnh, 211262 điểm)"
"[CẢNH BÁO] 188/388 ảnh có pose ... KHÔNG có file trong .../train/images — xoá khỏi reconstruction..."
"Xong. num_reg_images=200 num_points3D=211262"
```

Tôi đối chiếu 188 tên file "missing" đó với `test/images` và `train/images` cục bộ:

```python
missing = 188 tên trong log
missing ∩ test/images  = 50   # đúng bằng toàn bộ test set — hợp lý, test không dùng để train
missing ∩ train/images = 0
missing - test - train  = 138  # KHÔNG tồn tại ở BẤT KỲ ĐÂU trong public_set cục bộ
```

Tức là: sparse gốc từng đăng ký **388 ảnh** (chắc chắn là toàn bộ chuyến bay drone gốc), nhưng `public_set/hcm0031` chỉ phát hành **200 train + 50 test = 250/388 (64.4%)**. **138 khung hình (35.6%) có pose 3D thật nhưng ảnh không được phát hành cho benchmark này** (không phải bug restore của repo — `audit_round1_public_images.py`/`verify_round1_public_restore.py` chỉ kiểm tra kích thước file và tính duy nhất trong tập đã có, không đối chiếu với số ảnh gốc 388). Đây là một **giới hạn dữ liệu thật, không phải lỗi pipeline** — 200 train camera có khoảng cách baseline lớn hơn kỳ vọng ở nhiều đoạn quỹ đạo, và không "trick" huấn luyện nào bù đắp được chỗ dữ liệu vốn dĩ không có.

### 1.4. Diagnostic sẵn có mâu thuẫn với chính giả thuyết nền tảng của `B2`

`pipeline/work/hcm0031/diagnose_distance.csv` (còn sót lại từ script `09_diagnose_distance.py`, đã bị **xoá** ở commit `eb17653 Simplify repo...` — không còn cách nào tái tạo lại số này từ code hiện tại) đo tương quan Pearson giữa "khoảng cách tới camera train gần nhất" và chất lượng ảnh test. Tôi tự tính lại từ CSV:

```
corr(dist, psnr)  = +0.195
corr(dist, ssim)  = +0.387
corr(dist, lpips) = -0.347   (âm = xa hơn thì LPIPS THẤP hơn = TỐT hơn)
n = 50
```

Giả thuyết nền tảng cho `B2` (trích `09_diagnose_distance.py` docstring gốc): *"pose test nằm ở góc không có ảnh train gần → dễ sinh floaters ở vùng khuyết"*. Nếu đúng, kỳ vọng là khoảng cách **càng xa** thì chất lượng **càng tệ** (corr âm với ssim/psnr, dương với lpips). Số liệu thật lại **ngược chiều**: ảnh test xa camera train gần nhất lại có xu hướng **tốt hơn** ở mức trung bình-yếu (`r` 0.19–0.39, không mạnh nhưng đúng hướng ngược). Nhìn thô vào dữ liệu: cụm ảnh index `0174, 0189, 0199, 0200...` (giữa/cuối chuyến bay, `dist≈1.26–1.27`) đạt `tower-crop score 0.75–0.78`, còn cụm đầu chuyến bay (`0006–0125`, `dist≈0.1–0.9`) chỉ đạt `0.63–0.73`. Điều này gợi ý bottleneck thật có thể liên quan tới **một biến khác thay đổi theo thời gian bay** (góc nắng, độ cao, phần cảnh được chụp) nhiều hơn là "khoảng trống view train" — một giả thuyết mà **chưa ai kiểm chứng** trong repo, và nó làm suy yếu lý do chính đáng hoá `B2` mà `trick tham khảo.md` đưa ra ở mục 1.2–1.3.

### 1.5. `B2` đã chạy thật ngoài `pipeline/work/hcm0031/` và thất bại thảm khốc — 2 lần

`pipeline/work/hcm0031/colmap/dense/stereo/depth_maps` **rỗng** (0 file), không có `fused.ply` trong `pipeline/work/hcm0031/`, không có `04_colmap_dense_summary.txt` — đúng như những gì Codex quan sát (mục 1.6 của Codex). Nhưng có **hai** nơi khác trong repo chứa bằng chứng B2 đã chạy xong thật:

- `colab_b2_results_drop/` — có `fused.ply` (**121,495,051 bytes**), `04_patch_match_stereo.log` (1.7MB), `04_stereo_fusion.log`, `04_colmap_dense_summary.txt` — dense stereo **đã chạy thành công** ở một lần chạy khác.
- `downloads/B2_done.ipynb` và `downloads/B2_done 2 safe.ipynb` — hai notebook đã **train+render+eval thật** trên nguồn `prepared` (giống hệt baseline nhưng có patch cho depth args). Kết quả in ra (2 lần chạy độc lập, cùng pattern):

```
Cell 20 (B2_done.ipynb):   PSNR mean 10.5148  SSIM 0.3596  LPIPS 0.4233  Score 0.4016  (psnr_max=50, khớp baseline)
Cell 23 (B2_done 2 safe.ipynb): PSNR mean 10.5509  SSIM 0.3614  LPIPS 0.4226  Score 0.4027

Train log (cùng cell): "Training progress: 36%|... Loss=0.0766803, Depth Loss=0.0000000"
```

`PSNR ~10.5 dB` gần như là **nhiễu/vô nghĩa** (không phải "hơi tệ hơn", mà là hỏng), tái lặp ở 2 lần chạy độc lập với số gần như giống hệt nhau (`10.5148` và `10.5509`) — **không phải nhiễu ngẫu nhiên, mà là lỗi hệ thống có tính lặp lại**. Đồng thời `Depth Loss=0.0000000` xuyên suốt (giống hệt baseline) chứng tỏ **depth supervision chưa từng thực sự chạy** trong lần thử này nữa — nghĩa là cái được gọi "B2" ở đây thậm chí chưa test được điều nó dự định test.

**Điều này thay đổi hoàn toàn bức tranh so với `trao đổi.md`** (viết `2026-07-22 01:09`, mtime sớm hơn `B2_done.ipynb`, `2026-07-23 23:33`): tài liệu chốt hướng vẫn mô tả `B2` như bước "cần pilot để xác nhận chi phí" (`trick/README.md`: *"Nếu prepared không thắng full-image score 0.6731, dừng sớm trước khi mở thêm trick mới"*) — nhưng thực ra **đã có run thật cho thấy `prepared`/`B2` thua rất xa, không phải "chưa biết"**. Kết luận đúng ở thời điểm hiện tại phải là: dừng ngay chuỗi `B2 -> C/F -> A`, và trước khi thử lại bất kỳ biến thể `B2` nào, phải cách ly nguyên nhân sập điểm.

### 1.6. Nghi phạm cụ thể cho vụ sập điểm: cấu hình low-VRAM tự động tắt densification

`downloads/B2_done.ipynb`, cell 20, in log wrapper thật:

```
[b2-train-render-eval] LOW_VRAM_PROFILE=1
[b2-train-render-eval] RESOLUTION=4
```

Không cell nào trong notebook set `DENSIFY_*`/`LOW_VRAM_PROFILE` tường minh. Theo `pipeline/scripts/03_train_3dgs.sh` dòng 122–160:

```bash
if [[ "$LOW_VRAM_PROFILE" == "auto" ]]; then
  if [[ -d /content || -d /kaggle/working ]]; then LOW_VRAM_PROFILE="1"; else LOW_VRAM_PROFILE="0"; fi
fi
if [[ "$LOW_VRAM_PROFILE" == "1" ]]; then
  [[ "$RESOLUTION" == "-1" ]] && RESOLUTION="4"
  [[ -z "$DENSIFY_UNTIL_ITER" ]] && DENSIFY_UNTIL_ITER="0"
  [[ -z "$DENSIFY_FROM_ITER" ]] && DENSIFY_FROM_ITER="0"
  [[ -z "$DENSIFICATION_INTERVAL" ]] && DENSIFICATION_INTERVAL="0"
  [[ -z "$PERCENT_DENSE" ]] && PERCENT_DENSE="0.0"
  ...
fi
```

Nghĩa là mặc định, chạy trên `/content` (Colab) hoặc `/kaggle/working` sẽ tự động: ảnh còn **1/4 độ phân giải** VÀ **tắt hoàn toàn densification** (Gaussian không bao giờ nhân đôi/tách ra sau khi khởi tạo). So sánh: baseline `0.6731` chạy cũng trên `/kaggle/working` (theo path trong `03_train_3dgs.log`: `/kaggle/working/pipeline/work/hcm0031/gs_model`) nhưng **không** dính hiệu ứng này (ảnh gần full-res, point cloud tăng lên 1.23GB) — tức là ở lần chạy baseline, `LOW_VRAM_PROFILE`/`RESOLUTION` hẳn đã được set tường minh khác `auto`/`-1`, còn ở lần chạy B2 thì không. **Đây là một khác biệt cấu hình thực sự giữa hai lần chạy, độc lập với việc "dense source có tốt hay không"**, và là ứng viên số 1 để giải thích vụ sập PSNR xuống 10.5dB — cần bị loại trừ bằng thực nghiệm trước khi kết luận gì về giá trị của `prepared`/depth (xem mục 6, thí nghiệm A).

### 1.7. Bằng chứng "loại B1, hạ ưu tiên C" đến từ một dataset/benchmark khác hẳn

`git show 9383e23:WORKLOG.md`, `git show 8e88bc0:WORKLOG.md` xác nhận: `depth-prior=0.644`, `antenna-focus=0.6611` so với `baseline A=0.6616` đều đo trên **`HCM0421`**, **`MODE=holdout`**. `pipeline/common/scenes.py` (tại `eb17653^`) ghi rõ trong docstring:

> *"Round 2 (BTC đã bỏ round 1, thi lại từ đầu bằng dataset mới)... KHÔNG scene nào có ground-truth thật (`test/` chỉ có `test_poses.csv`)... muốn tự chấm điểm phải tự tạo holdout split từ chính ảnh train"*
> `BTS_SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]` — **`hcm0031` không nằm trong danh sách này.**

Tóm lại: bằng chứng "depth-prior thua, antenna-focus hoà" đến từ **Round 2** (dataset khác, tổ chức lại từ đầu), trên **scene khác** (`HCM0421` ≠ `hcm0031`), bằng **eval tự chế trên holdout train** (không phải GT thật). Thêm nữa, chính `WORKLOG.md` (`8e88bc0`) tự thừa nhận: kết quả `depth-prior=0.644` bị confound bởi phải hạ `SH_DEGREE 3→2` để né OOM ("không tách bạch được nguyên nhân thuần tuý"). Và kết luận về `antenna-focus` cũng tự giải thích là bị pha loãng vì Score tính trung bình trên 30 ảnh holdout, đa số không cận cảnh ăng-ten. `trao đổi.md` dòng 58–61 có ghi chú về confound `SH_DEGREE`, nhưng vẫn dùng cặp số này làm "bằng chứng thật duy nhất" để **tránh đề xuất lại `B1`/hạ kỳ vọng `antenna-focus` cho `hcm0031`** — đây là ngoại suy liên-scene, liên-round, liên-protocol chưa kiểm chứng, không nên có trọng số quyết định cho `hcm0031`.

### 1.8. Rủi ro đo lường thật đã tìm thấy trong chính repo

`downloads/B2_done.ipynb`, cell 26 (`MODE = "reeval_latest"`):

```python
PSNR_MAX = 30.0   # khác hẳn default 50.0 của pipeline/scripts/eval_round1_metrics.py
...
=== EVAL SUMMARY ===
PSNR mean : 10.5148   SSIM mean : 0.3596   LPIPS mean: 0.4233   Score mean: 0.4437
```

Với cùng `PSNR=10.5148`, `psnr_max=50` cho `Score=0.4016` (cell 20) còn `psnr_max=30` cho `Score=0.4437` (cell 26) — **chênh 0.042 điểm chỉ vì đổi hằng số chuẩn hoá**, không phải do model tốt hơn. Đây là ví dụ cụ thể, có thật, của **optimistic estimate/measurement bug** trong chính repo: nếu bất kỳ ai (kể cả người dùng, kể cả tôi hay Codex) vô tình lấy số `Score` từ nhánh `reeval_latest` này để so với baseline `0.6731` (chấm bằng `psnr_max=50`), kết luận sẽ sai ngay cả khi hai bên đều "PASS" quy trình.

### 1.9. Render-parity: rủi ro có thật, chưa đo được delta

`pipeline/scripts/render_round1_test_poses.py` dòng 40–44 và 156:

```python
class PipelineParamsStub:
    antialiasing = False        # default cứng trong stub
...
flags = read_pipeline_train_flags(model_dir)   # đọc model_dir/pipeline_train_flags.json
antialiasing = bool(flags.get("antialiasing", False))   # False nếu file không tồn tại
```

`pipeline_train_flags.json` **không tồn tại** trong `pipeline/work/hcm0031/gs_model/` cục bộ hiện tại (`find ... -iname "pipeline_train_flags*"` → rỗng), trong khi `trick/hcm0031/default.env` set `ANTIALIASING="1"` mặc định cho training. Nếu baseline `0.6731` từng được train với `--antialiasing` nhưng render lại **không** đọc được flag này (vì file bị mất/không đồng bộ về máy), số liệu đo được sẽ **không khớp đúng mô hình đã train** — một train/render mismatch âm thầm. Tôi không có đủ artifact cục bộ để đo delta thật (điểm này Codex đã nêu đúng ở mục 1.5 của họ, tôi xác nhận độc lập).

### 1.10. Câu hỏi tôi tự đặt ra trước khi đọc Codex: `0.85` có khả thi với 3DGS "vani" trên scene này không?

`pipeline/scripts/eval_round1_metrics.py`:

```python
def compute_score(psnr_v, ssim_v, lpips_v, psnr_max=50.0):
    psnr_norm = min(max(psnr_v / psnr_max, 0.0), 1.0)
    return 0.4*(1.0 - lpips_v) + 0.3*ssim_v + 0.3*psnr_norm
```

Đây là công thức **tuyến tính**, không có log-compression mạnh cho PSNR. Để đạt `Score=0.85` cần đồng thời khoảng: `LPIPS≈0.05` (đóng góp `0.38`), `SSIM≈0.90` (đóng góp `0.27`), `PSNR≈33-35dB` (đóng góp `0.20-0.21`). Mức chất lượng này tương đương SOTA 3DGS trên scene **bounded/indoor sạch** (Mip-NeRF360 indoor: PSNR ~31-33). Trên scene **outdoor/unbounded thật** như `hcm0031` (skyline mở, trụ mảnh, ảnh drone thật có nhiễu/exposure/motion blur), 3DGS/Mip-NeRF360 SOTA công bố thường chỉ đạt PSNR ~24-27dB, SSIM ~0.7-0.85, LPIPS ~0.2-0.35 — quy đổi ra Score theo công thức trên rơi vào khoảng **0.62-0.74**, thấp hơn hẳn `0.85`. Không có oracle nào trong repo (kể cả trước khi tôi đọc Codex) trả lời câu hỏi "trần khả thi trên `hcm0031` là bao nhiêu" — đây chính là điều tôi định đề xuất làm trước tiên, **trùng với "Oracle 1" của Codex** (xác nhận độc lập, xem mục 2 và 5).

---

## 2. Đọc `01_codex_proposal.md` — phản biện từng giả định

Sau khi đọc, các điểm Codex nêu ở mục `1.1–1.5, 1.7, 1.8` (baseline dùng `prepared` source, thiếu artifact, thiếu depth loss thật, rủi ro render-parity, dùng cẩn trọng bằng chứng `HCM0421`) **khớp với chẩn đoán độc lập của tôi** — tôi xác nhận các con số/trích dẫn Codex đưa ra (`cfg_args`, `eval_metrics.csv`, `eval_metrics_m0_tower_crop.csv`, `eval_metrics_m0_skyline_crop.csv`) là đúng, không có fabrication. Không đồng ý ở các điểm sau:

### 2.1. Codex bỏ sót bằng chứng lớn nhất trong repo: `B2` đã fail thật

Mục `1.6` của Codex: *"Dense stereo local hiện không có bằng chứng đầu ra thật... local state hiện không chứng minh được dense stereo đã chạy xong thành công."* Điều này **đúng nếu chỉ nhìn `pipeline/work/hcm0031/`**, nhưng **sai ở phạm vi repo đầy đủ**: `colab_b2_results_drop/fused.ply` (121MB, có thật) và đặc biệt `downloads/B2_done.ipynb`/`B2_done 2 safe.ipynb` chứa **kết quả train+eval thật** của chính xác thí nghiệm mà Codex đề xuất lại ở mục `5.1`/`Oracle 3.3` ("Raw vs Prepared vs Prepared+depths" / "Dense value oracle"). Hệ quả trực tiếp:

- Codex xếp thí nghiệm này vào **Tier 2** ("chi phí trung bình-cao, rủi ro trung bình", backlog #5, sau 4 mục Tier 1) — nhưng thực ra **không cần chạy lại từ đầu**: dữ liệu đã có sẵn, chỉ cần **phân tích lại log đã có** (rẻ hơn nhiều bậc so với xếp hạng của Codex).
- Codex viết: *"prepared_train_template... First real score gate against baseline"* (trích dẫn `experiment_matrix.csv`) như thể đây còn là việc "chưa làm" — trong khi thực tế **đã làm và thua rất xa** (`Score 0.40` vs `0.6731`). Không cập nhật fact này vào bảng ưu tiên là một lỗ hổng chẩn đoán quan trọng, vì nó thay đổi hẳn mức độ khẩn cấp: vấn đề không phải "chưa biết B2 có ích không", mà là "đã biết B2 (như đã implement) tệ hơn rất nhiều, cần điều tra NGUYÊN NHÂN trước khi làm gì khác liên quan tới `prepared`/depth".
- Codex cũng không phát hiện nghi phạm `LOW_VRAM_PROFILE=1`/`RESOLUTION=4`/tắt densify (mục 1.6 review này) — nghi phạm này đổi hẳn kết luận: có thể `B2` (dense/depth) tự nó không tệ, mà là **một bug cấu hình huấn luyện độc lập** đã phá hỏng lần thử đó. Codex vô tình đề xuất một thí nghiệm ("prepared+depths") có nguy cơ **lặp lại đúng lỗi cũ** và bị hiểu nhầm lần nữa là "depth không giúp".

### 2.2. Codex chấp nhận bằng chứng `HCM0421` mà không nêu confound Round 1 vs Round 2

Mục `1.8` và `2.1` của Codex trích `8e88bc0`, `9383e23`, `b696ff3` làm "facts" để kết luận *"không được đề xuất lại `depth-prior` kiểu cũ như thể chưa từng thử"* và *"không được coi `antenna-focus` là ứng viên mạnh"*. Đây là facts đúng **về những gì đã xảy ra trên `HCM0421`**, nhưng Codex không đối chiếu `pipeline/common/scenes.py` để thấy rằng `HCM0421` thuộc **Round 2** (dataset tổ chức lại hoàn toàn, không GT thật, eval bằng holdout tự chế), khác hẳn `hcm0031` (**Round 1** public, GT thật) — mức độ tin cậy khi ngoại suy sang `hcm0031` thấp hơn nhiều so với cách Codex trình bày ("facts" ngang hàng với các facts đo trực tiếp trên `hcm0031`). Tôi đề nghị hạ Level `1.8` từ "Facts" xuống "Inferences yếu, cross-domain, cross-protocol" trong khung phân loại Fact/Inference/Hypothesis mà chính Codex đề ra ở mục `2`.

### 2.3. Codex không thách thức tính khả thi của target `0.85`

Toàn bộ backlog của Codex (`Tier 1 → Tier 3`, `Oracle 1-3`) đều ngầm giả định rằng nếu dọn sạch đủ kỹ, một hướng nào đó trong danh sách sẽ đưa được score gần `0.85`. Codex có đề xuất `Oracle 1` (geometry-assisted IBR ceiling) đúng là công cụ để trả lời câu hỏi này — nhưng bản thân văn bản Executive Summary không đặt câu hỏi ngược: *nếu ngay cả oracle IBR cũng không gần `0.85`, liệu mục tiêu `0.85` bằng riêng 3DGS (không đổi radically representation/data) có còn hợp lý?* Câu hỏi này cần được nêu tường minh làm gate quyết định, không chỉ là một oracle âm thầm trong danh sách — vì nó ảnh hưởng tới việc có nên đầu tư GPU-giờ cho *bất kỳ* hướng nào trong `Tier 1/2` hay nên nhảy thẳng lên `Tier 3` (đổi representation / multi-model) sớm hơn khuyến nghị của Codex.

### 2.4. Codex đúng về "hygiene" nhưng đánh giá thấp phần "giới hạn dữ liệu cấu trúc"

Bottleneck #1 của Codex ("experiment hygiene") là đúng và tôi đồng ý xếp hạng cao. Nhưng ngay cả khi hygiene hoàn hảo (baseline vàng, checkpoint sweep đầy đủ, seed variance rõ, render parity đúng), phát hiện `1.3` của tôi (138/388 = 35.6% khung hình bị thiếu khỏi `public_set`) vẫn tồn tại độc lập — đây là **giới hạn dữ liệu**, không phải lỗi thực nghiệm, và Codex hoàn toàn không đề cập tới nó (không audit `01_run_colmap.log` theo hướng đối chiếu 388 vs 200+50). Hygiene tốt sẽ giúp đo đúng hơn, nhưng không tự động nâng trần dữ liệu.

### 2.5. Cả `trao đổi.md`/`trick tham khảo.md` và Codex đều "anchor" vào pipeline tuần tự — xem mục 4.

---

## 3. Lỗi đo lường / confound / leakage / optimistic estimate — tổng hợp

| # | Vấn đề | Bằng chứng | Ảnh hưởng |
|---|---|---|---|
| 1 | `PSNR_MAX` không nhất quán (`30.0` vs `50.0`) | `downloads/B2_done.ipynb` cell 26 `PSNR_MAX = 30.0`, so với default `50.0` trong `pipeline/scripts/eval_round1_metrics.py` | Cùng model, `Score` chênh `0.4016 → 0.4437` chỉ vì đổi hằng số — optimistic estimate có thật, không giả định |
| 2 | Bằng chứng loại `depth-prior`/`antenna-focus` từ scene/round/protocol khác | `9383e23`, `8e88bc0` trên `HCM0421` (Round 2, `MODE=holdout`), `pipeline/common/scenes.py` xác nhận Round 2 không có GT thật | Ngoại suy chưa kiểm chứng sang `hcm0031` (Round 1, GT thật) |
| 3 | `depth-prior=0.644` trên `HCM0421` tự confound bởi `SH_DEGREE 3→2` (fix OOM) | `8e88bc0` WORKLOG: *"không tách bạch được nguyên nhân thuần tuý"* (tự thừa nhận) | Kết luận "depth-prior thua" có thể do giảm SH, không phải do depth |
| 4 | Artifact cục bộ thiếu `iteration_30000`, `chkpnt*.pth`, `pipeline_train_flags.json` | `pipeline/work/hcm0031/gs_model/point_cloud/` chỉ có `iteration_15000`; render log baseline dùng `iteration_30000` | Không tự verify lại được `0.6731` từ máy hiện tại; render-parity antialiasing rủi ro thật |
| 5 | `diagnose_distance.csv` mồ côi — script sinh ra nó (`09_diagnose_distance.py`) đã bị xoá ở `eb17653` | `git log --all -- "*09_diagnose_distance*"` chỉ có 2 commit, không còn trong `pipeline/scripts/` hiện tại | Không audit lại được cách tính; corr đo được (mục 1.4) mâu thuẫn với giả thuyết nền tảng `B2` mà không ai kiểm tra lại |
| 6 | 138/388 khung hình gốc (35.6%) không tồn tại ở bất kỳ đâu trong `public_set/hcm0031` | Đối chiếu `01_run_colmap.log` (388 ảnh gốc, 188 bị loại) với `test/images` (50) và `train/images` (0 trùng) | `P0 PASS` (`count=200, unique=200`) tạo cảm giác dữ liệu "đủ sạch", nhưng không kiểm tra tính đầy đủ so với gốc — false confidence |
| 7 | `experiment_matrix.csv` gắn nhãn baseline là `source_mode=raw` | `trick/hcm0031/experiment_matrix.csv` dòng `baseline_ref` | Mâu thuẫn với `cfg_args` thật (`source_path=.../colmap/dense`) — bảng so sánh raw/prepared trong tương lai sẽ tự confound nếu dùng nhãn này làm control |
| 8 | `exposure.json` toàn identity-affine dù `EXPOSURE_COMP=1` là default | `pipeline/work/hcm0031/gs_model/exposure.json`, nhưng `cfg_args.train_test_exp=False` | Không rõ compensation có thật sự bật cho baseline hay không — cần audit trước khi coi baseline là "đã bật mọi cờ chuẩn" |
| 9 | Score cluster theo thời điểm bay, không chỉ theo khoảng cách camera | `eval_metrics_m0_tower_crop.csv`: cụm đầu (`0006–0125`) `0.63–0.73` vs cụm sau (`0174–0325`) `0.72–0.78` | Nguy cơ nhầm "cải thiện do trick X" với "cải thiện do trùng thời điểm/độ cao dễ hơn" nếu không kiểm soát theo cụm |
| 10 | Target `0.85` chưa có oracle nào xác nhận khả thi | Phân tích công thức `compute_score` (mục 1.10) | Rủi ro đốt GPU-giờ cho một trần không đạt được bằng 3DGS "vani" |

---

## 4. Anchoring vào pipeline `B2 -> C/F -> A`

`trao đổi.md` dòng 19: *"Các nhánh `E` (2 model / 2-stage) và `G` (đổi backbone) chỉ mở nếu pipeline trên không thắng rõ baseline trên `round1 public`."* `trick tham khảo.md` mục 4.1–4.3 lặp lại logic tương tự cho `2-model/blend`, `ensemble`, `đổi backbone` ("quá sớm... chưa cần khi B2 còn chưa kiểm chứng xong"). Ba vấn đề với khung này:

1. **Điều kiện mở khoá (`B2 -> C/F` "không thắng rõ") thực ra đã gần đạt** — theo mục `1.5` review này, `B2` (như đã implement) đã **thua rất xa**, không phải "chưa thắng rõ" mà là "thua thảm khốc". Nếu áp đúng logic mà `trao đổi.md` tự đặt ra, điều kiện mở `E`/`G` gần như đã được kích hoạt — chỉ là chưa ai cập nhật tài liệu để nhận ra điều đó (vì `trao đổi.md` viết trước khi `B2_done.ipynb` chạy).
2. **Không có ngưỡng định lượng nào cho "thắng rõ"** trong cả `trao đổi.md` lẫn `trick tham khảo.md` (chỉ có ngôn ngữ định tính "thắng rõ", "đủ tốt"). Codex có cải thiện điểm này bằng ngưỡng GO/STOP cụ thể (`±0.005`, `+0.01`...) — đây là điểm Codex làm tốt hơn tài liệu gốc, tôi ghi nhận.
3. **Cả 3 tài liệu (kể cả Codex) đều giả định con đường tối ưu là tuần tự (sequential gate)**, không phải song song. Với ngân sách GPU thực tế là **Kaggle/Colab free-tier** (phiên ~9-12h, T4/P100, VRAM hạn chế — bằng chứng: toàn bộ script phải có nhánh `LOW_VRAM_PROFILE`), chi phí "chờ" từng bước tuần tự (`B2` xong mới `C/F`, `C/F` xong mới `A`, tất cả xong mới `E`/`G`) có thể tốn **nhiều phiên hơn** so với chạy song song 2-3 hướng rẻ (checkpoint sweep, oracle ceiling, thử nhánh `feature/mip-splatting` có sẵn) trong cùng một khung thời gian. Bản chất bài toán (skyline mở-unbounded + trụ mảnh) là đúng loại bài toán mà **một Gaussian field toàn cục, loss đồng nhất** xử lý kém nhất theo chính tài liệu 3DGS/Mip-NeRF360 gốc — nhưng chưa có gì trong repo (kể cả đề xuất của Codex) thử nghiệm sớm liệu tách global/local hoặc đổi representation ngay từ đầu có rẻ hơn việc đi hết chuỗi tuần tự rồi mới "được phép" thử, nhất là khi `B2` đã cho thấy dấu hiệu cần nhiều vòng debug (đã fail 2 lần, nguyên nhân chưa rõ).

**Kết luận:** cả `trao đổi.md` và `01_codex_proposal.md` đều bị anchor vào "một model 3DGS toàn cục, cải tiến tuần tự" như trục duy nhất; sự khác biệt giữa hai bên chỉ là Codex thêm kỷ luật đo lường (đúng và cần thiết) chứ không thách thức trục này.

---

## 5. Phương án thay thế cho bước nhảy lớn tới `0.85`

### A. Cách ly nguyên nhân sập điểm của `B2` (ưu tiên tuyệt đối, phải làm trước mọi B/C/D bên dưới)
Chạy lại đúng nguồn `prepared` (không cần depth thật) nhưng **ép tường minh** `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`, không để `auto` tự tắt densify. So với baseline `0.6731`.

### B. Oracle trần khả thi (geometry-assisted view synthesis, không train gì)
Warp k-nearest train views (dùng `sparse`/`dense` COLMAP đã có) sang test pose, blend có occlusion-aware, chấm bằng đúng `eval_round1_metrics.py`. Nếu oracle này còn cách xa `0.85`, mục tiêu `0.85` bằng 3DGS thuần gần như không thực tế trong ngân sách hiện có — trùng ý `Oracle 1` của Codex, tôi xác nhận độc lập và đề xuất bổ sung: dùng **nhiều view lân cận + blend theo độ tin cậy depth**, không chỉ 1 view gần nhất (warp 1-view sẽ có lỗ hổng occlusion lớn, đánh giá thấp trần thật).

### C. Bật nhánh representation đã có sẵn trong repo: `feature/mip-splatting` / `feature/gsplat-mcmc`
```
remotes/origin/feature/mip-splatting
remotes/origin/feature/gsplat-mcmc
remotes/origin/feature/depth-anything-v2
remotes/origin/compact/compact-gaussian
```
Đây là **representation mới** với chi phí thấp hơn hẳn "đổi backbone from scratch" (Tier 3 của Codex) vì code đã tồn tại trong repo dưới dạng branch, không cần viết mới. `mip-splatting` (anti-aliased 3D Gaussian) đánh trực diện vào đúng bệnh "smear/floater vùng xa, ảnh drone góc rộng" mà `skyline-crop score 0.6384` chỉ ra — hợp lý hơn thử "2DGS/Zip-NeRF" hoàn toàn xa lạ.

### D. Model chuyên biệt cho far-field bằng sky/segmentation prior (rẻ, độc lập với `B2`)
Sky mask 2D (không cần GT test) loại bỏ Gaussian floaters ở vùng trời rõ ràng — một trick rẻ, không phụ thuộc vào `B2` có sửa được hay không, đánh trực tiếp vào `skyline-crop` mà không cần dense stereo.

### E. Local tower/antenna specialist + compositing hợp lệ (theo đúng ranh giới Codex mục `9`)
Train model B chỉ tối ưu vùng trụ (dùng bootstrap mask từ `trick/scripts/04_bootstrap_tower_masks.sh`), blend với model global bằng alpha cố định theo mask/geometry train-time — không fit theo GT test. Chỉ mở sau khi `masked tower eval` (không phải bbox crop) xác nhận trụ thật sự là bottleneck độc lập với nền.

### F. Multi-seed ensemble hợp lệ (uncertainty-weighted, không dùng GT test)
Mở rộng `03_checkpoint_sweep.sh` thành multi-seed, blend bằng trọng số nghịch đảo độ lệch giữa các model (proxy uncertainty, tính từ chính các render, không chạm GT test) — hạ tầng gần như đã có sẵn (`trick/scripts/03_checkpoint_sweep.sh`), chi phí tăng thêm chủ yếu là GPU-giờ train thêm N-1 seed.

### G. Local tower model độc lập hoàn toàn (crop-train riêng cho vùng gần trụ)
Thay vì compositing 2 model full-scene, train một model 3DGS **chỉ trên ảnh/crop gần trụ** (dùng `tower_bbox3d.json` đã có) với độ phân giải cao hơn, densify mạnh hơn — rồi paste vào model global bằng depth-aware compositing. Rủi ro artifact ở biên cao hơn E, nhưng có thể đạt chất lượng cục bộ cao hơn nhiều nếu vùng trụ đúng là bottleneck xác nhận.

### H. Camera/pose refinement thận trọng trên train poses
Vì 138/388 khung gốc bị thiếu (mục 1.3), 200 camera train còn lại có baseline lớn hơn kỳ vọng ở một số đoạn quỹ đạo — bundle-adjustment nhẹ / pose jitter optimization (nếu `GS_REPO` hỗ trợ, ví dụ nhánh có trong repo) có thể giảm sai số hình học tích luỹ.

### I. Score-oriented loss: thêm perceptual/LPIPS-term vào training loss
Công thức chấm điểm gán trọng số **cao nhất cho LPIPS (0.4)**, cao hơn SSIM (0.3) và PSNR (0.3), nhưng loss mặc định của 3DGS (`L1 + D-SSIM`) không tối ưu trực tiếp LPIPS. Thêm perceptual loss (LPIPS-based, dùng thư viện `lpips` đã có sẵn trong pipeline eval) vào hàm loss huấn luyện là một "score hack" hợp lệ (tối ưu đúng cái được chấm điểm) và tương đối rẻ để thử.

### J. Data-side: xin bổ sung 138 frame còn thiếu (đòn bẩy tiềm năng lớn nhất, nhưng ngoài tầm kiểm soát kỹ thuật)
Nếu có đường xin lại đủ 388 ảnh gốc từ nguồn phát hành `public_set`, đây là cách lấp gap quỹ đạo camera **thật** thay vì thuật toán cố bù đắp — headroom tiềm năng lớn hơn bất kỳ trick thuần thuật toán nào, nhưng phụ thuộc vào việc có nguồn dữ liệu sạch hơn hay không (đúng như "Trường hợp 2" trong `trao đổi.md` dòng 122–141, nhưng ở đó chỉ nói tới 4 scene còn thiếu của `public_set`, chưa nói tới 138 frame thiếu trong chính `hcm0031`).

---

## 6. Thí nghiệm rẻ nhất để bác bỏ/xác nhận từng phương án

| Phương án | Thí nghiệm rẻ nhất | Chi phí | Kết quả bác bỏ | Kết quả xác nhận |
|---|---|---|---|---|
| A. Cách ly bug low-VRAM | Rerun `prepared` source, `LOW_VRAM_PROFILE=0`, `RESOLUTION=-1`, giữ nguyên mọi thứ khác | 1 run ~30-60 phút GPU | `Score` vẫn sập gần `0.4x` → lỗi nằm ở nguồn `prepared`/depth, không phải cấu hình | `Score` phục hồi về vùng gần baseline → xác nhận bug cấu hình là thủ phạm chính, `B2` chưa bị loại |
| B. Oracle trần khả thi | Warp k-NN train view → test pose, chấm bằng `eval_round1_metrics.py`, không train | Vài giờ CPU/GPU nhẹ, không cần train 3DGS | Oracle cũng chỉ đạt `~0.65-0.72` → `0.85` không khả thi cho hướng thuần refine 3DGS, cần data/representation mới | Oracle đạt `>0.80` → còn nhiều headroom, đáng đầu tư tiếp cho refine hình học |
| C. `feature/mip-splatting` | Checkout branch, train 1 lần trên `hcm0031` cùng config baseline, so `skyline-crop`/`full-image` | 1 run train | Không đổi hoặc tệ hơn baseline | `Score` (đặc biệt `skyline-crop`) tăng rõ so với baseline |
| D. Sky mask cho far-field | Chạy sky segmentation (model nhẹ, off-the-shelf) trên 200 train + 50 test, tính lại `skyline-crop` sau khi mask floaters | Vài chục phút, không cần train lại | Ảnh hưởng không đáng kể lên `skyline-crop` | `skyline-crop` cải thiện rõ mà `full-image` không tụt |
| E. Local tower specialist + compositing | Train model B chỉ trên crop trụ 5-10k iter (ngắn), blend thử trên vài ảnh anchor có trụ | Trung bình (1 short train + compositing script) | `full-image` tụt hoặc biên trụ sinh artifact rõ | `full-image` tăng hoặc giữ nguyên, vùng trụ nét hơn theo cả metric lẫn mắt |
| F. Multi-seed ensemble | 2 seed thêm (ngoài baseline) cùng config, blend đơn giản trung bình pixel, so `full-image` | Trung bình-cao (2 train run đầy đủ) | Blend không thắng seed tốt nhất | Blend thắng cả seed tốt nhất |
| G. Local tower model riêng | Train riêng crop trụ độ phân giải cao, so PSNR/SSIM/LPIPS trên vùng mask trụ so với model global | Trung bình | Không cải thiện trên mask trụ thật | Cải thiện rõ trên mask trụ thật |
| H. Camera refinement | Bật pose optimization nhẹ (nếu có flag trong `GS_REPO`), so `full-image` | Thấp-trung bình | Không đổi hoặc bất ổn | `Score` tăng, đặc biệt ở ảnh test gần khu vực có gap 138 frame |
| I. LPIPS-aware loss | Thêm term LPIPS nhỏ vào loss, train ngắn (7-15k iter) so baseline cùng iter | Thấp-trung bình | `Score` không đổi hoặc `PSNR` tụt mạnh (trade-off xấu) | `LPIPS`/`Score` cải thiện mà `PSNR` không tụt quá `0.5dB` |
| J. Xin thêm 138 frame | Kiểm tra xem nguồn gốc dataset (organizer/GDrive/B2) có sẵn 388 ảnh đầy đủ không | Rất thấp (chỉ tra cứu) | Không có nguồn nào khác | Có nguồn đầy đủ hơn → ưu tiên lại toàn bộ roadmap |

---

## 7. Backlog xếp hạng theo Expected gain / độ tin cậy / GPU-hour / rủi ro

| # | Hạng mục | Expected gain | Độ tin cậy kết luận | GPU-hour | Rủi ro |
|---|---|---|---|---|---|
| 1 | **A. Rerun `prepared` với `LOW_VRAM_PROFILE=0` tường minh** (cách ly bug densify/resolution) | Cao — có thể "cứu" toàn bộ hướng `B2` chỉ bằng 1 dòng config, hoặc loại nó dứt khoát | Rất cao (thí nghiệm nhị phân, control tốt) | Thấp (~1 run) | Thấp |
| 2 | **B. Oracle trần khả thi** (warp view, không train) | Rất cao về mặt quyết định (định hình lại toàn bộ roadmap, kể cả có nên theo đuổi `0.85` bằng 3DGS thuần hay không) | Cao nếu implement đúng (occlusion-aware) | Rất thấp | Thấp |
| 3 | **Audit render-parity + rebuild gold baseline artifact đầy đủ** (đồng ý với Codex #1) | Trung bình-cao nếu có bug thật (`+0.01` trở lên theo ngưỡng của Codex) | Cao | Thấp (1 run sạch) | Thấp |
| 4 | **Checkpoint sweep dày** (đồng ý với Codex #2, đã có script) | Trung bình-cao, rẻ | Cao | Thấp | Thấp |
| 5 | **Real masked tower eval** (đồng ý với Codex #4, đã có script `04/05`) | Trung bình (định hướng ưu tiên, không tự nó tăng score) | Cao | Thấp-trung bình | Thấp |
| 6 | **C. Thử nhánh `feature/mip-splatting` có sẵn** | Trung bình-cao, đánh đúng bệnh skyline | Trung bình (chưa test lần nào trên `hcm0031`) | Trung bình (1-2 run) | Trung bình (khả năng breaking changes vs pipeline hiện tại) |
| 7 | **D. Sky-mask cho far-field** | Trung bình, độc lập rủi ro với B2 | Trung bình-cao | Thấp | Thấp |
| 8 | **I. LPIPS-aware training loss** (score-oriented) | Trung bình, tối ưu trực tiếp cái được chấm | Trung bình | Thấp-trung bình | Trung bình (rủi ro trade-off PSNR/LPIPS) |
| 9 | **3 seed variance** (đồng ý Codex #3) | Trung bình (chất lượng quyết định, không phải điểm số) | Cao | Trung bình | Thấp |
| 10 | **H. Camera/pose refinement nhẹ** | Trung bình, đánh vào đúng gap 138 frame | Trung bình | Thấp-trung bình | Trung bình |
| 11 | **`prepared + true depth`** (chỉ SAU khi #1 loại trừ xong bug config) | Trung bình nếu #1 xác nhận bug config là thủ phạm; thấp nếu #1 vẫn sập | Trung bình (phụ thuộc #1) | Trung bình-cao | Trung bình |
| 12 | **F. Multi-seed ensemble hợp lệ** | Trung bình-cao nếu oracle blend (7.2 style) cho thấy gain lớn | Trung bình | Cao (N lần train đầy đủ) | Trung bình |
| 13 | **E. Local tower specialist + compositing** | Trung bình-cao, chỉ nếu masked eval (#5) xác nhận trụ là bottleneck riêng | Trung bình | Trung bình-cao | Trung bình-cao |
| 14 | **G. Local tower model độ phân giải cao riêng** | Cao cục bộ, nhưng ảnh hưởng lên `full-image score` chưa chắc lớn (bài học từ `antenna-focus` trên `HCM0421` — dù khác scene, cơ chế pha loãng average vẫn áp dụng) | Thấp-trung bình | Trung bình-cao | Cao (artifact biên, compositing phức tạp) |
| 15 | **J. Xin bổ sung 138 frame còn thiếu** | Tiềm năng cao nhất nếu khả thi, nhưng phụ thuộc yếu tố ngoài kỹ thuật | Không xác định trước (phụ thuộc nguồn dữ liệu có tồn tại hay không) | Rất thấp để kiểm tra khả năng | Thấp để thử, cao nếu phải re-run toàn bộ pipeline với data mới |
| 16 | **Đổi representation triệt để (2DGS/Zip-NeRF-class từ đầu)** | Không xác định — chỉ nên cân nhắc nếu Oracle B cho thấy trần 3DGS quá thấp | Thấp cho tới khi có Oracle B | Rất cao | Rất cao |

**Khuyến nghị thứ tự thực thi:** `#1 → #2` (song song, cả hai đều rẻ và quyết định-quan trọng) `→ #3, #4, #5` (dọn hygiene, song song với nhau) `→` dựa trên kết quả `#1`/`#2` để chọn nhánh `#6-#11` `→` chỉ mở `#12-#14` sau khi có tín hiệu định lượng từ các bước trên, và chỉ cân nhắc `#16` nếu Oracle (#2) cho thấy trần 3DGS thuần rõ ràng thấp hơn `0.85` một cách có ý nghĩa.
