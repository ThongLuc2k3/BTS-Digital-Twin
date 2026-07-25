# 01 Codex Proposal

## Executive Summary

Mục tiêu nâng score từ khoảng `0.6731` lên ít nhất `0.85` là một bước nhảy rất lớn. Với bằng chứng hiện có trong repo, không có cơ sở để tin rằng chỉ cần tuning nhẹ hoặc tiếp tục đúng pipeline hiện tại là sẽ đạt được mức đó. Nút thắt lớn nhất lúc này không phải là thiếu ý tưởng trick, mà là thiếu kỷ luật thực nghiệm: baseline hiện tại chưa được đóng gói đầy đủ, checkpoint tốt nhất chưa được xác minh, parity train/render chưa chắc đúng, và nhiều kết luận trong tài liệu vẫn chưa được lấp bằng run thật.

Kết luận chiến lược:

- Không mặc định pipeline hiện tại là đúng.
- Không mở rộng nhánh trick mới trước khi dựng lại một baseline vàng có artifact đầy đủ.
- `B2` vẫn đáng giữ, nhưng chỉ còn ý nghĩa nếu chứng minh được `true depth supervision` thắng baseline hiện hành; vì baseline local hiện đã train từ `prepared dense source`.
- Các hướng local tower, antenna-focus, depth-prior cũ phải bị nghi ngờ mặc định do đã có bằng chứng thất bại hoặc bug thật trong lịch sử commit.

## 1. Verified Facts

### 1.1 Score và metric hiện có

Các số sau là dữ kiện thật, xác minh trực tiếp từ file trong repo:

- `full-image`: `PSNR 21.6938`, `SSIM 0.6819`, `LPIPS 0.1542`, `Score 0.6731`
  - nguồn: [pipeline/work/hcm0031/eval_metrics.csv](../pipeline/work/hcm0031/eval_metrics.csv)
- `tower-crop`: `PSNR 23.2912`, `SSIM 0.7287`, `LPIPS 0.1298`, `Score 0.7064`
  - nguồn: [pipeline/work/hcm0031/eval_metrics_m0_tower_crop.csv](../pipeline/work/hcm0031/eval_metrics_m0_tower_crop.csv)
- `skyline-crop`: `PSNR 20.4286`, `SSIM 0.6298`, `LPIPS 0.1829`, `Score 0.6384`
  - nguồn: [pipeline/work/hcm0031/eval_metrics_m0_skyline_crop.csv](../pipeline/work/hcm0031/eval_metrics_m0_skyline_crop.csv)

### 1.2 Baseline local hiện tại không phải raw baseline thuần

Model local hiện tại được train từ:

- `source_path='/kaggle/working/pipeline/work/hcm0031/colmap/dense'`
  - nguồn: [pipeline/work/hcm0031/gs_model/cfg_args](../pipeline/work/hcm0031/gs_model/cfg_args)

Đây là dữ kiện rất quan trọng: baseline `0.6731` trong workspace này đã hưởng lợi từ `prepared dense source`, không còn là baseline raw sparse thuần như cách tài liệu đôi lúc mô tả.

### 1.3 Baseline hiện tại chưa dùng true depth supervision

Trong log train local:

- `Depth Loss=0.0000000` xuyên suốt run
  - nguồn: [pipeline/work/hcm0031/03_train_3dgs.log](../pipeline/work/hcm0031/03_train_3dgs.log)

Suy ra:

- dense source đã được dùng
- nhưng depth regularization thật chưa được dùng

### 1.4 Artifact thực nghiệm local đang thiếu

Trong `pipeline/work/hcm0031/gs_model`, hiện chỉ thấy:

- `cfg_args`
- `cameras.json`
- `events.out...`
- `exposure.json`
- `input.ply`
- `point_cloud/iteration_15000`

Thiếu:

- `pipeline_train_flags.json`
- `chkpnt30000.pth`
- `point_cloud/iteration_30000/point_cloud.ply`

Hệ quả:

- resume không đáng tin
- checkpoint selection gần như chưa làm được
- render parity có thể sai

### 1.5 Render parity có nguy cơ đang sai

`render_round1_test_poses.py` đọc antialiasing từ:

- `model_dir/pipeline_train_flags.json`

Nếu file này không có, nó mặc định:

- `antialiasing = False`

Trong khi `03_train_3dgs.sh` lại train với:

- `ANTIALIASING=1`
- `EXPOSURE_COMP=1`

và chỉ ghi `pipeline_train_flags.json` sau khi train xong.

Điều này tạo một failure mode nguy hiểm:

- train một kiểu
- render/eval một kiểu khác

### 1.6 Dense stereo local hiện không có bằng chứng đầu ra thật

Local workspace có các thư mục:

- `pipeline/work/hcm0031/colmap/dense/stereo/depth_maps`
- `.../normal_maps`
- `.../consistency_graphs`

nhưng hiện tại số file trong cả ba là:

- `0`

Ngoài ra local không có:

- `logs/04_patch_match_stereo.log`
- `logs/04_stereo_fusion.log`
- `logs/04_colmap_dense_summary.txt`

Nghĩa là local state hiện không chứng minh được dense stereo đã chạy xong thành công.

### 1.7 Ma trận thí nghiệm hiện gần như chưa có run thật

File [trick/hcm0031/experiment_matrix.csv](../trick/hcm0031/experiment_matrix.csv) chỉ có:

- `baseline_ref`
- các row template `planned`

Thiếu hoàn toàn run thật cho:

- `prepared_train`
- `checkpoint_sweep`
- `masked tower eval`
- `true depth supervision`

### 1.8 Commit history bác bỏ vài hướng nếu lặp lại y nguyên

Các commit cũ cung cấp bằng chứng thật:

- `8e88bc0`: `DEPTH_PRIOR=1` thua baseline trên `HCM0421`
- `9383e23`: `antenna-focus` gần như hòa baseline
- `b696ff3`: antenna pipeline từng có bug scale rất nặng

Suy ra:

- không được đề xuất lại `depth-prior` kiểu cũ như thể chưa từng thử
- không được coi `antenna-focus` là ứng viên mạnh nếu chưa có chẩn đoán mới

## 2. Distinguishing Facts, Inferences, Hypotheses

### 2.1 Facts

- baseline `0.6731` là thật
- tower-crop > full-image, skyline-crop < full-image là thật
- local baseline được train từ `colmap/dense`
- local baseline không dùng depth loss thật
- local artifacts đang thiếu
- local dense stereo evidence đang thiếu
- experiment matrix chưa có run thật ngoài baseline

### 2.2 Inferences

- narrative hiện tại có xu hướng đánh giá cao `B2` hơn mức dữ kiện thật đang hỗ trợ
- trần hiện tại có thể đang bị “ảo thấp” do hygiene thực nghiệm kém
- nhiều claim về chất lượng pipeline hiện hành chưa đủ chuẩn để đưa ra quyết định GPU lớn

### 2.3 Unverified hypotheses

- chỉ sửa render parity và chọn checkpoint đúng có thể lấy thêm điểm đáng kể
- multi-seed có thể lộ variance đủ lớn để thay đổi thứ hạng run
- true depth supervision có thể vẫn giúp thêm, dù prepared source đã được baseline dùng
- bottleneck tower thật có thể nhỏ hơn cảm giác nếu masked eval cho thấy bbox crop đang đánh giá sai

## 3. Bottlenecks

### 3.1 Bottleneck số 1: experiment hygiene

Đây là bottleneck chắc chắn nhất hiện tại.

Biểu hiện:

- thiếu checkpoint 30k
- thiếu pipeline flags
- thiếu dense logs
- thiếu checkpoint sweep
- thiếu run matrix

Nếu không sửa chỗ này trước:

- mọi kết luận kỹ thuật đều có khả năng sai gốc

### 3.2 Bottleneck số 2: skyline / far-field

Đây là bottleneck được hỗ trợ tốt nhất bởi metric hiện có:

- `skyline-crop score 0.6384 < full-image 0.6731`

Đây là bằng chứng phù hợp với nhận định:

- nền xa
- skyline
- floater
- smear

là lỗi toàn cục đáng xử lý.

### 3.3 Bottleneck số 3: tower metric hiện tại chưa đủ tin

`tower-crop` đang cao hơn `full-image`, nhưng đây mới là bbox crop, không phải mask pixel thật.

Do đó hiện chưa thể kết luận chắc rằng:

- tower đã ổn
hoặc
- tower là bottleneck chính

Phải có masked tower eval thật mới được kết luận.

### 3.4 Bottleneck số 4: missing oracle on achievable ceiling

Hiện chưa có thí nghiệm nào trả lời câu hỏi:

- nếu hình học và view interpolation đã tốt hơn nhiều thì score tối đa có thể lên đến đâu?

Không có oracle này, rất dễ đốt GPU vào hướng không thể đưa score đến gần `0.85`.

## 4. Strategy Tier 1: Safe, High ROI

### 4.1 Rebuild one gold baseline package end-to-end

Mục tiêu:

- tạo một baseline chuẩn có đủ artifact
- score reproducible
- train/render parity rõ ràng

File/code bị ảnh hưởng:

- [pipeline/scripts/03_train_3dgs.sh](../pipeline/scripts/03_train_3dgs.sh)
- [pipeline/scripts/render_round1_test_poses.py](../pipeline/scripts/render_round1_test_poses.py)
- [pipeline/scripts/manage_b2_artifacts.py](../pipeline/scripts/manage_b2_artifacts.py)

Chi phí:

- 1 run GPU sạch

Rủi ro:

- thấp

Chỉ số phải đo:

- `full-image score`
- artifact completeness:
  - `chkpnt30000.pth`
  - `iteration_30000/point_cloud.ply`
  - `pipeline_train_flags.json`

Ngưỡng GO/STOP:

- GO nếu score tái hiện trong `±0.005`
- STOP mọi trick mới nếu chưa dựng xong baseline vàng

### 4.2 Mandatory checkpoint sweep

Mục tiêu:

- biết checkpoint nào tốt nhất theo `full-image score`

File/code:

- [trick/scripts/03_checkpoint_sweep.sh](../trick/scripts/03_checkpoint_sweep.sh)

Chi phí:

- thấp

Rủi ro:

- thấp

Chỉ số:

- best `full-image score`
- score curve theo iteration

Ngưỡng GO/STOP:

- nếu checkpoint tốt nhất > checkpoint cuối ít nhất `0.01`, từ đó trở đi checkpoint sweep là bắt buộc
- nếu chênh rất nhỏ `<0.003`, không cần tốn công sweep quá dày

### 4.3 Multi-seed on exact same baseline

Mục tiêu:

- đo variance thật

File/code:

- [pipeline/scripts/03_train_3dgs.sh](../pipeline/scripts/03_train_3dgs.sh)

Chi phí:

- trung bình

Rủi ro:

- thấp

Chỉ số:

- mean score
- std score

Ngưỡng GO/STOP:

- nếu `std > 0.005`, không được tin bất kỳ kết luận single-run nào
- nếu `std < 0.003`, có thể ra quyết định bằng 1-2 run nhanh hơn

### 4.4 Real masked tower evaluation

Mục tiêu:

- phân biệt tower thật với nền ăn theo trong bbox crop

File/code:

- [trick/scripts/04_bootstrap_tower_masks.sh](../trick/scripts/04_bootstrap_tower_masks.sh)
- [trick/scripts/05_run_m0_mask_eval.sh](../trick/scripts/05_run_m0_mask_eval.sh)
- [trick/scripts/eval_round1_mask_metrics.py](../trick/scripts/eval_round1_mask_metrics.py)

Chi phí:

- thấp đến trung bình

Rủi ro:

- thấp

Chỉ số:

- masked tower score
- delta giữa masked tower và bbox tower

Ngưỡng GO/STOP:

- nếu masked tower tụt rất mạnh so với bbox tower, tower vẫn là bottleneck thật
- nếu masked tower vẫn ổn, ưu tiên background/far-field trước

### 4.5 Audit render parity explicitly

Mục tiêu:

- loại bỏ khả năng chấm sai do render flags sai

File/code:

- [pipeline/scripts/render_round1_test_poses.py](../pipeline/scripts/render_round1_test_poses.py)
- [pipeline/scripts/03_train_3dgs.sh](../pipeline/scripts/03_train_3dgs.sh)

Chi phí:

- thấp

Rủi ro:

- thấp

Chỉ số:

- score difference giữa render có antialiasing vs không

Ngưỡng GO/STOP:

- nếu parity fix đem lại `+0.01` hoặc hơn, coi đây là bugfix bắt buộc trước mọi trick khác

## 5. Strategy Tier 2: Bold Pipeline / Model Changes

### 5.1 Raw vs Prepared vs Prepared+Depths

Đây là thí nghiệm quyết định để biết `B2` còn headroom thật hay không.

So sánh công bằng:

1. `raw`
2. `prepared`
3. `prepared + true --depths`

File/code:

- [pipeline/scripts/04_run_colmap_dense.sh](../pipeline/scripts/04_run_colmap_dense.sh)
- [pipeline/scripts/03_train_3dgs.sh](../pipeline/scripts/03_train_3dgs.sh)
- [pipeline/scripts/generate_b2_variant_notebooks.py](../pipeline/scripts/generate_b2_variant_notebooks.py)

Chi phí:

- trung bình đến cao

Rủi ro:

- trung bình

Chỉ số:

- `full-image score`
- `skyline-crop score`
- stability

Ngưỡng GO/STOP:

- nếu `prepared + depths` không thắng best non-depth baseline ít nhất `0.01`, dừng depth branch

### 5.2 Hard-view mining / short curriculum resume

Ý tưởng:

- lấy nhóm view tệ nhất
- fine-tune ngắn để giảm lỗi ở các view đó

Đây là nhánh gần với tinh thần `error-guided refine`, nhưng phải đo bằng `full-image`, không phải nhìn ảnh.

File/code bị ảnh hưởng:

- training wrapper hiện tại
- có thể mượn hướng từ commit `e2d7d60`

Chi phí:

- trung bình

Rủi ro:

- trung bình

Chỉ số:

- worst-10 view score
- full-image score

Ngưỡng GO/STOP:

- nếu worst-view tăng nhưng full-image giảm quá `0.005`, stop

### 5.3 Global/background model + local/tower model

Đây là hướng hai thành phần nhưng vẫn còn trong vùng chấp nhận được nếu không dùng test leakage.

Nguyên tắc:

- model A tối ưu background / skyline
- model B tối ưu tower / thin structure
- compositing phải dựa trên geometry, mask, ROI có sẵn từ train-time, không được fit theo GT test

File/code:

- render pipeline
- eval pipeline
- một compositing step mới

Chi phí:

- cao

Rủi ro:

- trung bình cao

Chỉ số:

- full-image score
- skyline score
- masked tower score

Ngưỡng GO/STOP:

- chỉ tiếp tục nếu từng nhánh đơn lẻ đã thắng ở metric vùng của nó

### 5.4 Pose / exposure / color refinement

Hiện đây chưa phải hướng được chứng minh là bottleneck chính, nhưng có thể là boost trung hạn.

Ứng viên:

- exposure consistency
- color calibration nhẹ
- nếu có thể, camera refinement rất thận trọng trên train poses

File/code:

- training repo clone
- render/eval parity

Chi phí:

- trung bình

Rủi ro:

- trung bình

Chỉ số:

- LPIPS toàn ảnh
- skyline score
- view-to-view consistency

Ngưỡng GO/STOP:

- nếu LPIPS giảm mà score tổng không tăng, dừng

## 6. Strategy Tier 3: Crazy but Potentially Large-Jump

### 6.1 Representation swap

Ví dụ:

- 2DGS
- scaffold-based GS
- hybrid mesh + Gaussian
- Zip-NeRF-class alternative

File/code:

- gần như stack mới

Chi phí:

- rất cao

Rủi ro:

- rất cao

Chỉ số:

- `full-image score` trên cùng benchmark

Ngưỡng GO/STOP:

- chỉ mở nếu oracle cho thấy 3DGS hiện tại có ceiling thấp rõ rệt

### 6.2 Heterogeneous render-space ensemble

Không phải ensemble ngây thơ.

Cần:

- các model có lỗi bổ sung nhau thật
- blending rule cố định, hợp lệ, không dựa vào GT test

File/code:

- compositor mới
- audit legal boundary

Chi phí:

- cao

Rủi ro:

- cao

Chỉ số:

- score gain so với best single model

Ngưỡng GO/STOP:

- nếu blend oracle dưới `+0.01`, dừng

### 6.3 Explicit specialist for thin structures

Hướng “điên rồ nhưng có logic”:

- chuyên gia riêng cho tower / antenna / wire
- kết hợp với global model bằng mask/geometry

File/code:

- training and merge stack mới

Chi phí:

- rất cao

Rủi ro:

- cao

Chỉ số:

- masked tower score
- full-image score

Ngưỡng GO/STOP:

- chỉ làm nếu masked tower thực sự là deficit lớn nhất sau Tier 1

## 7. Oracle Experiments

### 7.1 Oracle 1: Geometry-assisted IBR ceiling

Mục tiêu:

- đo trần tiềm năng của dữ liệu và hình học trước khi đốt GPU vào model mới

Ý tưởng:

- dùng train images + COLMAP geometry
- warp nearest views sang held-out views
- chấm score

Nếu oracle này vẫn rất xa `0.85`, thì:

- mục tiêu `0.85` với pipeline hiện tại gần như không thực tế

### 7.2 Oracle 2: Candidate blend upper bound

Mục tiêu:

- biết ensemble/compositing có đáng thử không

Ý tưởng:

- lấy 2-3 candidate render hợp lệ
- blend bằng rule không học theo GT test
- đo gain

Nếu gain rất nhỏ:

- bỏ ensemble sớm

### 7.3 Oracle 3: Dense value oracle

Mục tiêu:

- biết `B2` còn giá trị riêng bao nhiêu

Ý tưởng:

- chạy công bằng:
  - `raw`
  - `prepared`
  - `prepared + depths`

Nếu `prepared` đã cho phần lớn gain, còn `depths` gần như không thêm gì:

- dừng đầu tư sâu vào depth branch

## 8. Full Consideration of Requested Axes

### 8.1 Multi-seed and HPO

- phải làm multi-seed trước khi tin bất kỳ delta nhỏ nào
- HPO nên bắt đầu cực hẹp:
  - `sh_degree`
  - `densify_grad_threshold`
  - save/checkpoint schedule
  - low-vram overrides

Không mở grid lớn khi baseline còn chưa đóng gói sạch.

### 8.2 Checkpoint selection

Đây là hướng ROI cao nhất chưa được làm đúng.

Hiện local package còn thiếu cả checkpoint 30k, nên kết luận về “checkpoint cuối” gần như chưa có giá trị.

### 8.3 Ensemble / mixture / compositing

Có thể hợp lệ, nhưng chỉ nếu:

- blending rule không dùng GT test
- vùng phối hợp được xác định từ geometry / prior / mask sinh từ train

### 8.4 Global/background + local/tower split

Đây là hướng hợp lý hơn local trick đơn lẻ nếu:

- skyline và tower thật sự là hai failure mode khác nhau
- hai model có thể bổ sung sai số cho nhau

### 8.5 Backbone / representation replacement

Có thể là hướng cuối cùng nếu mục tiêu `0.85` là nghiêm túc.

Khoảng cách `0.6731 -> 0.85` quá lớn để ngây thơ tin vào tuning nhẹ.

### 8.6 Pose / camera / exposure / color refinement

Nên để sau khi:

- baseline reproducible
- checkpoint tốt nhất đã rõ

Nếu không, rất khó phân biệt gain do refinement hay do nhiễu thực nghiệm.

### 8.7 Thin structure / skyline / far-field

Đây vẫn là hai bệnh được hỗ trợ tốt nhất bởi metric:

- skyline: có bằng chứng thật
- thin structure: có nghi ngờ mạnh nhưng thiếu masked metric thật

### 8.8 Hard-view mining / curriculum / loss / densification

Đáng thử nhưng nên đi sau:

- baseline vàng
- checkpoint sweep
- masked tower eval

### 8.9 Mọi trick trong `trick/`

Đánh giá thực tế:

- `01_dense_pilot.sh`: hợp lý nhưng local evidence chưa đủ
- `02_prepared_train.sh`: currently redundant if baseline local already used prepared source
- `03_checkpoint_sweep.sh`: nên được đẩy lên bắt buộc
- `04/05 mask eval`: rất đáng làm vì giúp quyết định hướng tower

## 9. Legal / Leakage Boundary

Loại bỏ các hướng sau:

- bất kỳ tuning nào fit trực tiếp theo GT test image
- bất kỳ blend/compositing nào chọn trọng số theo GT test
- bất kỳ mask hoặc region prior nào được rút từ test GT thay vì geometry/train-time prior

Chỉ chấp nhận:

- dùng `round1 public` như benchmark đối chiếu vì repo hiện đang làm đúng điều đó
- dùng geometry, sparse/dense, train images, tower OBB, bootstrap masks sinh từ train-time assets

## 10. Recommended Backlog by Expected Gain / Cost / Risk

### Top priority

1. Rebuild one gold baseline package with full artifacts and verified train/render parity.
   - Expected gain: high confidence, possible immediate score recovery
   - Cost: low
   - Risk: low

2. Run checkpoint sweep on a denser save schedule and compare best checkpoint to current reported baseline.
   - Expected gain: medium to high
   - Cost: low
   - Risk: low

3. Run 3 seeds on exact same config.
   - Expected gain: medium in decision quality
   - Cost: medium
   - Risk: low

4. Produce real masked tower evaluation.
   - Expected gain: medium in prioritization quality
   - Cost: low to medium
   - Risk: low

### Next priority

5. Run clean `raw vs prepared vs prepared+depths` experiment.
   - Expected gain: medium
   - Cost: medium to high
   - Risk: medium

6. Try hard-view short resume only if worst-view diagnosis supports it.
   - Expected gain: medium
   - Cost: medium
   - Risk: medium

7. Try narrow HPO after sweep and seed variance are understood.
   - Expected gain: low to medium
   - Cost: medium
   - Risk: medium

### Only if prior stages justify it

8. Global/background + local/tower compositing.
   - Expected gain: medium to high
   - Cost: high
   - Risk: medium high

9. Representation swap / heterogeneous ensemble.
   - Expected gain: potentially high
   - Cost: very high
   - Risk: high

## Final Recommendation

Điều nguy hiểm nhất lúc này là tiếp tục tin rằng mình đang tối ưu model, trong khi thực tế còn chưa biết baseline đúng là gì. Trước khi theo đuổi mục tiêu `0.85`, repo này cần một baseline vàng có artifact đầy đủ, checkpoint sweep thật, seed variance thật, và một phép đo tower đúng hơn bbox crop. Chỉ sau khi dọn sạch phần đó, mọi kết luận về `B2`, depth supervision, hay local refine mới có giá trị kỹ thuật.

Nếu phải chọn một câu chốt:

- hướng khả dĩ nhất để tăng nhanh không phải “thêm trick mới”, mà là “khử trần giả do pipeline và thực nghiệm thiếu sạch”, rồi mới dùng các thí nghiệm phân tách `raw / prepared / prepared+depths` để biết `B2` còn đáng đầu tư đến đâu.
