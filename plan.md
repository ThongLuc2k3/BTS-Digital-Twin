# Kế hoạch duy nhất — Restart với dữ liệu Round 2

## 0. Thay đổi bối cảnh quan trọng nhất

- Vòng thi cũ (dataset `VAI_NVS_DATA`, 13 scene, deadline 30/07, đã nộp thật đạt
  `58.67320`) **đã bị BTC bỏ**. Cuộc thi bắt đầu lại từ dữ liệu mới
  `Dataset/VAI_NVS_DATA_ROUND2/`.
- Deadline mới, số scene chính thức cần nộp, và có còn `public_set` để tự chấm hay không
  đều **chưa được BTC xác nhận lại bằng văn bản** trong repo này — cần hỏi kênh hỗ trợ
  chính thức ngay. Cho tới khi có xác nhận, giả định xấu nhất: **7 scene hiện có trong
  `VAI_NVS_DATA_ROUND2/` chính là toàn bộ tập phải nộp, không có ảnh ground-truth nào**,
  và thời gian còn lại ngắn.
- Quy định thi (`Đề bài.md` mục 9): *"Dữ liệu và scene hoàn toàn mới được cung cấp cho
  mỗi vòng thi, cách thức tính điểm sẽ được giữ nguyên."* → công thức chấm điểm, luật
  chống gian lận, giới hạn hạ tầng vẫn giữ nguyên như vòng cũ. Phần lớn hạ tầng kỹ thuật
  (branches, script, phân tích nguyên nhân điểm thấp) đã xây ở vòng trước **vẫn dùng lại
  được**, không phải làm lại từ đầu.

## 1. Ràng buộc bắt buộc phải tuân thủ (Đề bài.md mục 10)

Đọc kỹ trước khi làm bất kỳ ý tưởng "không giới hạn" nào, vì có ít nhất 1 cái bẫy rất dễ
dính với dữ liệu round 2 cụ thể:

- **Cấm dùng dữ liệu ngoài chứa cùng đối tượng/scene của đề thi**, cấm dùng bất kỳ nguồn
  nào để tái tạo/suy luận ground-truth của tập test.
  - ⚠️ Bẫy cụ thể: scene `bonsai` trong `VAI_NVS_DATA_ROUND2/` trùng tên với scene chuẩn
    nổi tiếng của bộ dataset học thuật **Mip-NeRF 360** (có ảnh gốc + test GT công khai
    trên mạng). **Tuyệt đối không tải bộ Mip-NeRF360 `bonsai` gốc về dùng làm GT tự chấm
    hoặc dùng làm dữ liệu train/fine-tune bổ sung** — dù ảnh có giống hệt hay không, đây
    là vi phạm rõ ràng mục 10.1. Chỉ được dùng đúng ảnh train + sparse COLMAP mà BTC cấp
    trong `VAI_NVS_DATA_ROUND2/bonsai/`.
  - Pretrained weight của model nền tảng tổng quát (Depth Anything V2, v.v., không train
    trên chính scene thi) vẫn hợp lệ — đã xác nhận ở vòng trước, giữ nguyên diễn giải này.
- **Mọi ảnh nộp phải do thuật toán sinh 100% tự động** — không chỉnh sửa tay từng ảnh,
  không ghép/vẽ/xoá vật thể thủ công. Bước hậu xử lý (mục 5 dưới) phải là code chạy hàng
  loạt, không phải thao tác tay từng pose.
- **Phải tái lập được**: nếu vào top, phải nộp được code, config, version thư viện,
  checkpoint, training log. Giữ pin commit hash cho mọi repo ngoài (đã làm ở vòng trước
  với `gaussian-splatting` và `Depth-Anything-V2`, tiếp tục giữ chuẩn này).
- Số lượng model **không giới hạn**, được ensemble, được fine-tune tự do — đây là chỗ
  "ý tưởng không giới hạn" của bạn có đất dùng hợp lệ, xem mục 6.

## 2. Dữ liệu Round 2 — tóm tắt (chi tiết đầy đủ đã ghi ở lượt trao đổi trước)

```
Dataset/VAI_NVS_DATA_ROUND2/
├── HCM0421/  ├── HCM0539/  ├── HCM0540/  ├── HCM0644/  ├── HCM0674/   (BTS, drone)
├── bonsai/                                                            (tổng quát, indoor)
└── chair/                                                             (tổng quát, portrait)
```

| Scene | Loại | Train img | Test pose | Resolution | Scale | GT test? |
|---|---|---|---|---|---|---|
| HCM0421/0539/0540/0644/0674 | BTS | 240 | 60 | 1320×989 | 1/4 | Không |
| bonsai | tổng quát | 248 | 28 | 1920×1080 | 1/1 | Không |
| chair | tổng quát | 205 | 58 | 720×1280 (dọc) | 1/1.5 | Không |

Điểm phải xử lý khác vòng trước:

1. **Không scene nào có ảnh GT** → không thể tự chấm Score theo kiểu cũ (so ảnh render
   với `test/images/`). Phải tự tạo bộ validation nội bộ (mục 4).
2. **2/7 scene không phải BTS** (`bonsai`, `chair`) → mọi kỹ thuật đặc thù BTS (antenna-
   focus, giả định cấu trúc mảnh kim loại) phải **tắt/không áp dụng** cho 2 scene này,
   không được coi là mặc định toàn cục.
3. **`chair` là ảnh dọc (portrait), độ phân giải/scale khác nhau giữa các scene** → mọi
   chỗ code hard-code `1320x989` hoặc giả định ảnh ngang phải kiểm tra lại, đọc
   `width/height/fx/fy` trực tiếp từ CSV (nguyên tắc này đã được ghi đúng từ vòng trước,
   chỉ cần xác nhận lại code thật sự tuân theo, không giả định).
4. Format `sparse/0/` (rig-format COLMAP mới: `frames.bin`, `rigs.bin`) giống hệt vòng
   trước → code đọc COLMAP hiện tại (`pipeline/common/colmap_runner.py`) nhiều khả năng
   chạy thẳng được, chỉ cần test lại trên 1 scene mới trước khi tin tưởng toàn bộ.

## 3. Di sản kỹ thuật giữ lại — không làm lại từ đầu

Toàn bộ phân tích nguyên nhân điểm thấp ở `Hướng đi.md` vẫn còn giá trị (áp dụng nguyên
cho 5 scene BTS, áp dụng một phần cho `bonsai`/`chair`):

- (a) cấu trúc mảnh (ăng-ten/dây cáp) → Gaussian phình to → mờ. *(chỉ áp dụng BTS)*
- (b) pose test ở góc thiếu ảnh train → floaters vùng khuyết. *(áp dụng mọi scene)*
- (c) nền chiếm phần lớn khung hình, lãng phí ngân sách Gaussian. *(áp dụng mọi scene,
  mức độ khác nhau)*

Code/branch đã có sẵn, tái sử dụng được ngay:

| Nhánh/script | Trạng thái | Dùng cho round 2 thế nào |
|---|---|---|
| `main` — vanilla 3DGS (Inria) | Baseline đã chạy thật, ổn định | Baseline bắt buộc cho mọi scene, kể cả `bonsai`/`chair` |
| `feature/mip-splatting` — antialiasing, đã fix bug lệch train/render | Code xong, đã chạy nhưng chưa benchmark sạch | Ứng viên mặc định Stage 1, generic — áp dụng mọi scene |
| `feature/depth-anything-v2` — depth prior 16-bit | Code xong, chưa benchmark sạch | Generic (depth monocular không đặc thù BTS) — thử trên mọi scene, ưu tiên cao vì đánh trực tiếp vào nguyên nhân (b) |
| `07_build_antenna_weights.py` / `apply_antenna_patch.py` (antenna-focus) | Code có sẵn, patch chưa xác nhận áp sạch lên commit mới | **Chỉ dùng cho 5 scene BTS**, tắt hẳn với `bonsai`/`chair` |
| `feature/gsplat-mcmc` | Code chạy được, chưa cho thấy thắng rõ `main` đã fix bug | Giữ làm phương án ensemble/so sánh per-scene (mục 6), không phải trục chính |
| `compact/compact-gaussian` | Chưa có bằng chứng GPU thật | Tạm bỏ, không đủ thời gian debug thêm hướng rủi ro |
| `Y_TUONG_TRR_HAU_XU_LY_THAM_CHIEU.md` — ý tưởng TRR (hậu xử lý tham chiếu hình học) | Mới ở mức thiết kế, **chưa code** | **Đây chính là ý tưởng "chạy lại bằng train data" bạn vừa đề xuất — nâng lên thành Stage 2 bắt buộc, xem mục 6.2** |

## 4. Phương pháp đánh giá bắt buộc — thay đổi cốt lõi so với vòng trước

Vòng trước dựa vào `public_set` có ảnh GT thật để tự chấm. Vòng này **không scene nào có
GT**, nên không được benchmark bằng cảm quan "nhìn đẹp hơn". Cách duy nhất hợp lệ và không
vi phạm luật (không đụng dữ liệu ngoài):

**Tự tạo tập validation nội bộ từ chính ảnh train được cấp**, theo từng scene:

1. Với mỗi scene, giữ lại ngẫu nhiên ~10–15% ảnh train (giữ đa dạng góc nhìn, không lấy
   toàn ảnh liền kề nhau) làm `holdout` — coi như "test có GT" nội bộ.
2. Train generator trên phần còn lại (~85–90% ảnh train).
3. Render đúng pose của các ảnh `holdout` (dùng pose thật đọc từ `images.bin`, không phải
   `test_poses.csv` — vì đó là ảnh train, đã có pose sẵn).
4. Tính PSNR/SSIM/LPIPS/Score như bình thường so với ảnh `holdout` thật.
5. Dùng điểm này để **chọn cấu hình/model thắng cho từng scene** (mục 6.3).
6. **Sau khi đã chọn xong cấu hình thắng**, train lại lần cuối trên **100% ảnh train**
   (không giữ holdout nữa) rồi mới render `test_poses.csv` thật để nộp bài — không lãng
   phí 10–15% dữ liệu ở bản nộp cuối.

Quy tắc bắt buộc khi so sánh (giữ nguyên tinh thần vòng trước):

- Đơn vị so sánh là Score trung bình trên **nhiều scene** (ít nhất 3 scene đại diện: 1 BTS
  dễ, 1 BTS khó/nhiều floaters, `chair` hoặc `bonsai`), không quyết định từ 1 scene.
  Với thời gian gấp, có thể co lại còn 2 scene (1 BTS + 1 non-BTS) cho vòng lặp đầu, mở
  rộng ra đủ 7 scene trước khi chốt cấu hình cuối.
- Không ra quyết định bằng mắt.
- `PSNR_max=50` (quy ước nội bộ giải ngược từ điểm chấm thật vòng trước) tiếp tục dùng
  làm chuẩn nội bộ cho tới khi có thông tin khác.

## 5. Kiến trúc pipeline — trả lời trực tiếp yêu cầu "1 model ra kết quả, hoặc lấy kết quả + train data chạy lại cho tốt hơn"

Chốt kiến trúc **2 tầng**, đúng như ý tưởng TRR đã thiết kế sẵn nhưng chưa code:

```
Ảnh train scene
      │
      ▼
[STAGE 1] Generator chính (3DGS họ Inria, mip-splatting + depth prior)
      │   → render ảnh RGB tại test_poses.csv
      ▼
Ảnh render thô (có thể mờ/floaters ở vùng thiếu view)
      │
      ▼
[STAGE 2] TRR Tier-1 — Hậu xử lý tham chiếu hình học (KHÔNG train model mới)
      │   input: ảnh render Stage 1 + ảnh train thật + sparse COLMAP (pose đã biết)
      │   1. với mỗi pose test, tìm k ảnh train gần nhất (theo vị trí + góc nhìn camera)
      │   2. dùng sparse points3D + pose để xác định điểm 3D nào nhìn thấy được từ ảnh
      │      train đó và từ pose test đó (visibility qua COLMAP, không đoán mò)
      │   3. warp/project texture thật từ ảnh train lên đúng vị trí trong ảnh render
      │   4. blend theo trọng số: góc nhìn, khoảng cách, occlusion, độ tin cậy tái chiếu
      ▼
Ảnh cuối cùng để nộp
```

Vì sao đây là lựa chọn đúng cho "không giới hạn ý tưởng" trong khuôn khổ luật thi:

- Không hallucinate — texture lấy đúng từ ảnh train thật đã biết pose, không "vẽ thêm"
  gì mới → rủi ro làm giảm PSNR/SSIM/LPIPS thấp hơn nhiều so với diffusion/enhancer tổng
  quát (đã phân tích và loại ở `Y_TUONG_TRR_HAU_XU_LY_THAM_CHIEU.md` mục 3).
- Không vi phạm mục 10.1/10.4: không dùng dữ liệu ngoài, hoàn toàn tự động, không cần
  clone thêm pretrained weight mới nào.
- Tận dụng đúng hạ tầng đã có (`pipeline/common/poses.py`, `alignment.py`).
- Generic — áp dụng được cho cả `bonsai`/`chair`, không đặc thù BTS.
- Effort thấp hơn nhiều so với đổi trainer (`gsplat-mcmc`) hay thêm model diffusion.

**Tier-2 (tuỳ chọn, chỉ làm nếu Tier-1 đo có lợi rõ và còn dư thời gian):** với vùng hoàn
toàn khuất khỏi mọi ảnh train (Tier-1 bó tay), cân nhắc model chuyên dụng sửa artifact
3DGS thưa view (ví dụ 3DGS-Enhancer, pretrained, không train trên data thi) — rủi ro cao
hơn Tier-1 nên **không phải việc làm ngay**, chỉ làm nếu Phase B/C dưới xong sớm.

## 6. Chọn model Stage 1 theo từng scene — dùng đúng quyền "ensemble không giới hạn"

Vì đề cho phép không giới hạn số model, chiến lược thực dụng nhất với thời gian gấp:

6.1. Với **5 scene BTS**: thử 2 cấu hình trên tập holdout (mục 4) —
  - `A` = mip-splatting antialiasing-only (mặc định, đã fix bug)
  - `B` = `A` + depth prior (Depth Anything V2)
  - Nếu có thời gian dư: `C` = `B` + antenna-focus
  - Chọn cấu hình thắng theo Score holdout **riêng từng scene** (không bắt buộc 5 scene
    BTS phải dùng chung 1 cấu hình nếu 1 scene cụ thể phản ứng khác).

6.2. Với `bonsai`/`chair` (non-BTS): chỉ thử `A` và `B` (không có antenna-focus). Nếu
  `gsplat-mcmc` rảnh tay để thử thêm (đã có code sẵn), thử làm `D` để so sánh — đặc biệt
  đáng thử với `bonsai` vì là scene indoor dày Gaussian, ngân sách MCMC có thể có lợi thế
  thật (khác với kết luận "chưa thắng" trên scene BTS ở vòng trước — đó là kết luận riêng
  cho domain BTS, không tự động đúng cho domain khác).

6.3. Stage 2 (TRR Tier-1) áp dụng **sau khi** đã chọn xong Stage 1 thắng cho từng scene,
  đo tiếp trên holdout xem TRR có thật sự tăng Score không trước khi áp vào bản nộp cuối
  (tránh lặp lại sai lầm vòng trước: đừng ship ý tưởng chưa đo được).

## 7. Roadmap thực thi rút gọn (ưu tiên vì thời gian gấp)

Không làm tuần tự cứng theo "ngày" vì chưa biết deadline mới — làm theo thứ tự giá trị/
rủi ro, chốt xong bước nào nộp thử bước đó (miễn đã qua threshold), không dồn hết vào
cuối.

**Phase 0 — Hạ tầng tối thiểu (bắt buộc, chặn mọi việc sau)**
- [ ] Xác nhận lại deadline/số scene chính thức với BTC qua kênh hỗ trợ
- [ ] Test `pipeline/common/colmap_runner.py` đọc sparse của 1 scene BTS mới + `chair`
      (ảnh dọc) + `bonsai` — đảm bảo không hard-code resolution/aspect ratio ở đâu
- [ ] Sửa `06_package_submission.py` để nhận đúng tên 7 scene mới (không phải số nguyên
      dạng `HCM0xxx` cố định — cần chạy được với tên chữ thường `chair`/`bonsai`)
- [ ] Viết script tạo holdout split (mục 4) — chọn ảnh, tách folder, sinh pose GT nội bộ

**Phase A — Baseline mọi scene**
- [ ] Chạy cấu hình `A` (mip-splatting antialiasing-only) trên toàn bộ 7 scene, đo Score
      holdout, lưu vào `Kết quả/` với tên rõ ràng gắn round2

**Phase B — Depth prior**
- [ ] Chạy cấu hình `B` trên toàn bộ 7 scene, so với `A` bằng Score holdout
- [ ] Chốt `A` hay `B` theo từng scene

**Phase C — TRR Tier-1**
- [ ] Code script hậu xử lý hình học (mục 5), chạy sau bước render
- [ ] Đo trên holdout của 2–3 scene đại diện (1 BTS, 1 non-BTS) trước khi bật cho toàn bộ
- [ ] Bật cho toàn bộ scene nếu đo có lợi, tắt cho scene nào đo không có lợi

**Phase D — Antenna-focus (chỉ 5 scene BTS, chỉ nếu còn thời gian)**
- [ ] Xác nhận `apply_antenna_patch.py` áp sạch lên commit hiện tại
- [ ] Đo trên holdout scene BTS, chỉ giữ nếu tăng rõ

**Phase E — Chốt & nộp**
- [ ] Với mỗi scene, train lại lần cuối trên 100% ảnh train bằng cấu hình đã thắng
- [ ] Render `test_poses.csv` thật, chạy TRR Tier-1 nếu đã xác nhận có lợi
- [ ] Đóng gói, kiểm tra checklist mục 8, **nộp sớm** — không chờ tới hạn chót mới nộp
      bản đầu tiên
- [ ] Lặp lại Phase D/Tier-2 nếu còn thời gian sau khi đã có 1 bản nộp an toàn

**Việc chủ động bỏ qua để giữ tốc độ (có thể quay lại nếu rất dư thời gian):**
- Edge loss (#4 vòng trước) — lợi ích thấp-trung bình, không phải ưu tiên khi gấp
- Appearance/exposure embedding (#5) — chỉ bật nếu quan sát rõ lệch màu ở 1 scene cụ thể
- TRR Tier-2 (3DGS-Enhancer) — chỉ làm nếu Tier-1 đã đo có lợi và còn rất dư thời gian
- `compact/compact-gaussian` — rủi ro cao, không đủ effort để debug trong thời gian gấp

## 8. Checklist trước mỗi lần nộp (giữ nguyên từ vòng trước, cập nhật số scene)

- [ ] Đủ đúng số thư mục scene theo xác nhận mới nhất từ BTC (hiện thấy 7: HCM0421,
      HCM0539, HCM0540, HCM0644, HCM0674, bonsai, chair) — tên thư mục thật, không phải
      `scene_001`
- [ ] Mỗi scene: số ảnh = số dòng `test_poses.csv` scene đó (không đều nhau giữa scene)
- [ ] Mỗi ảnh đúng `width×height` đọc từ CSV, đặc biệt chú ý `chair` (720×1280, ảnh dọc —
      dễ bị code cũ giả định ngang làm sai kích thước)
- [ ] Tên file giữ nguyên đúng theo cột `image_name` trong `test_poses.csv`
- [ ] Zip không chứa thư mục rác (`__MACOSX/`...)
- [ ] Test giải nén lại ở thư mục sạch để chắc đúng cấu trúc yêu cầu

## 9. Việc không được làm

- Không tải/dùng bất kỳ bản sao nào của dataset `bonsai` (hay bất kỳ scene nào khác)
  ngoài đúng file BTC cấp trong `VAI_NVS_DATA_ROUND2/` — kể cả để "chỉ xem thử" cũng
  không nên tải về máy đang dùng cho pipeline thi.
- Không dùng diffusion/enhancer tổng quát để "vẽ lại đẹp hơn" (rủi ro hallucinate sai chi
  tiết thật, làm giảm điểm dù ảnh nhìn nét hơn) — nếu cần "model thứ 2", đi theo TRR.
- Không đánh giá/quyết định bằng 1 scene đơn lẻ hoặc bằng mắt.
- Không áp antenna-focus cho `bonsai`/`chair`.
- Không chờ đến sát hạn chót mới nộp bản đầu tiên — nộp sớm ngay khi có bản qua được
  baseline, sau đó nộp đè bản tốt hơn.
