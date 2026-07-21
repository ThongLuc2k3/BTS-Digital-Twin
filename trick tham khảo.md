# Trick tham khảo cho BTS Digital Twin

File này không liệt kê mẹo chung chung. Mục tiêu là:

- bám đúng khó khăn đã thấy trên `hcm0031`
- bám đúng số liệu đang có trong repo
- chọn các `trick` có khả năng tăng `full-image score`
- và mô tả cách dựng từng `trick` từ số 0 tới lúc chấm được

## 1. Tình hình hiện tại cần nhớ trước khi làm trick

### 1.1. Baseline đang có trên `hcm0031`

Theo các CSV đã lưu trong `pipeline/work/hcm0031`:

- `full-image`: `PSNR 21.6938`, `SSIM 0.6819`, `LPIPS 0.1542`, `Score 0.6731`
- `tower-crop`: `PSNR 23.2912`, `SSIM 0.7287`, `LPIPS 0.1298`, `Score 0.7064`
- `skyline-crop`: `PSNR 20.4286`, `SSIM 0.6298`, `LPIPS 0.1829`, `Score 0.6384`

### 1.2. Bệnh thật của scene

Các lỗi đã thống nhất:

- `background xa / skyline` bị `floater`, smear, nhiễu
- `trụ BTS / anten / dây mảnh` bị ghosting, blur, vỡ cấu trúc
- `train/test gap` có thật, nên bài toán không chỉ là train chưa đủ lâu

### 1.3. Điều này dẫn tới kết luận gì

- `B2` là bước hợp lý nhất để xử lý `background xa`
- `C/F` chỉ nên làm sau `B2`, không nên nhảy vào refine cục bộ quá sớm
- `checkpoint selection` và `tuning nhẹ` vẫn đáng làm, nhưng không thể thay cho `B2`

## 2. Top 5 trick nên thử theo đúng thứ tự cho `hcm0031`

Đây là thứ tự ưu tiên thực dụng nhất hiện tại.

### Trick 1. `B2 dense-stereo pilot`

#### Vì sao trick này hợp với số liệu đang có

- `skyline-crop` đang tệ hơn `full-image` rõ rệt
- lỗi nhìn bằng mắt và lỗi đo bằng số cùng chỉ vào `background xa`
- đây là đúng chỗ `COLMAP dense-stereo` có cơ hội giúp nhiều nhất

#### Mục tiêu thắng

- chạy được `patch_match_stereo + stereo_fusion`
- có `fused.ply`
- có log thời gian và biết chi phí thật
- không lỗi runtime/OOM

#### Cách làm từ số 0 đến cuối

1. Dùng notebook Colab:
- `pipeline/kaggle_b2_pilot_hcm0031.ipynb`

2. Giữ mode mặc định:
- `RUN_DENSE = '1'`
- `RUN_TRAIN = '0'`
- `RUN_RENDER = '0'`
- `RUN_EVAL = '0'`

3. Chạy tới cuối.

4. Lấy về:
- `b2_pilot_artifacts.zip`

5. Kiểm tra trong zip:
- `04_colmap_dense_summary.txt`
- `04_patch_match_stereo.log`
- `04_stereo_fusion.log`
- `fused.ply` nếu có

#### Điều kiện pass

- `stereo_fusion` không fail
- `fused.ply` sinh ra được
- thời gian chạy đủ chấp nhận để còn lặp nhiều test

#### Điều kiện fail

- runtime không ổn định
- depth map gần như rỗng
- `fused.ply` lỗi hoặc quá nghèo

#### Nếu pass thì làm gì tiếp

- sang `Trick 2`

#### Nếu fail thì làm gì tiếp

- chưa đụng `C/F`
- sửa `B2` pilot trước: cache, max image size, GPU index, dataset runtime

---

### Trick 2. Train hoặc resume với `SOURCE_MODE=prepared`

#### Vì sao trick này hợp

Sau `Trick 1`, ta đã có dense workspace ở:

- `pipeline/work/hcm0031/colmap/dense`

Nếu source này sạch hơn raw sparse thì đây là cách rẻ nhất để xem `B2` có chuyển hóa được thành tăng điểm hay không.

#### Mục tiêu thắng

- train/resume chạy được với source `prepared`
- render lại `round1 public`
- `full-image score` tăng so với baseline `0.6731`

#### Cách làm từ số 0 đến cuối

1. Chuẩn bị môi trường có:
- `GS_REPO`
- checkpoint/model hiện tại nếu resume

2. Chạy:

```bash
export SOURCE_MODE=prepared
bash pipeline/scripts/03_train_3dgs.sh hcm0031
```

3. Render lại test poses.

4. Chấm lại `full-image`.

5. So với baseline cũ.

#### Điều kiện pass

- `full-image score` tăng thật
- không chỉ nhìn ảnh thấy đẹp hơn

#### Điều kiện fail

- chỉ đổi source nhưng điểm tổng không tăng
- hoặc train mất ổn định hơn bản raw

#### Ghi chú quan trọng

- đây là `trick` chuyển tín hiệu `B2` thành training effect
- nếu dense stereo chỉ đẹp trên log mà không giúp điểm, phải biết dừng sớm

---

### Trick 3. Quét checkpoint thay vì lấy checkpoint cuối

#### Vì sao trick này hợp

Đây là trick rẻ nhất nhưng rất hay có lời. Với bài này, khác biệt nhỏ ở `LPIPS` hoặc `SSIM` có thể đổi score cuối.

#### Mục tiêu thắng

- chọn được checkpoint tốt hơn checkpoint mặc định
- tăng điểm mà không cần đổi kiến trúc

#### Cách làm từ số 0 đến cuối

1. Giữ nhiều checkpoint trung gian:
- ví dụ `15000`, `30000`, `45000`, `60000`

2. Với mỗi checkpoint:
- render test poses
- chấm `full-image`

3. Ghi bảng so sánh:
- checkpoint
- `PSNR`
- `SSIM`
- `LPIPS`
- `Score`

4. Chọn checkpoint có `Score` tốt nhất, không chọn theo cảm tính.

#### Điều kiện pass

- có ít nhất một checkpoint thắng baseline hoặc thắng checkpoint cuối

#### Điều kiện fail

- mọi checkpoint dao động rất ít hoặc đều tệ hơn baseline

#### Ghi chú quan trọng

- đây là trick bắt buộc phải có trong mọi test về sau
- vì nếu không, bạn có thể đánh giá sai cả `B2`

---

### Trick 4. Tuning nhẹ theo scene `hcm0031`

#### Vì sao trick này hợp

Sau khi đã có baseline rõ, dense source rõ, checkpoint selection rõ, lúc đó tuning nhỏ mới có giá trị.

#### Các biến nên đụng trước

- `iterations`
- `densify_grad_threshold`
- `sh_degree`

#### Các biến chưa nên đụng trước

- quá nhiều cờ train cùng lúc
- mọi trick cục bộ cho tower
- thay backbone

#### Cách làm từ số 0 đến cuối

1. Chỉ mở một bảng test nhỏ.

2. Mỗi lần chỉ đổi một biến chính.

3. Luôn giữ:
- cùng scene
- cùng source
- cùng cách render
- cùng cách chấm

4. Ghi kết quả theo bảng:
- config
- full-image score
- ghi chú quan sát hình

#### Điều kiện pass

- có một config tăng điểm rõ hơn baseline control

#### Điều kiện fail

- tăng crop cảm tính nhưng score tổng không tăng
- quá nhiều biến đổi cùng lúc, không biết cái gì hiệu quả

---

### Trick 5. Local refinement cho vùng trụ chỉ sau khi `B2` đã giúp nền xa

#### Vì sao trick này hợp

`tower-crop` hiện đang cao hơn `full-image`, nhưng điều này chưa chứng minh trụ đã tốt; crop vẫn lẫn nhiều nền dễ. Vì vậy refine cục bộ vẫn hợp lý, nhưng chỉ sau khi nền xa đỡ bẩn hơn.

#### Mục tiêu thắng

- giảm ghosting/blur ở trụ
- không phá nền đã được cải thiện bởi `B2`
- tốt nhất là làm tăng `full-image score`, không chỉ tăng crop

#### Cách làm từ số 0 đến cuối

1. Chốt một baseline sau `B2`.

2. Chọn vùng trụ cần ưu tiên:
- ROI box
- hoặc danh sách ảnh test/train nhìn rõ trụ

3. Áp dụng một trick cục bộ duy nhất trước:
- `ROI emphasis`
- hoặc `weighted loss`
- hoặc `fine-tune ngắn`

4. Train ngắn.

5. Render/chấm lại.

6. So lại với model sau `B2` nhưng trước local refine.

#### Điều kiện pass

- hình vùng trụ tốt hơn
- và `full-image score` không bị tụt

#### Điều kiện fail

- crop có vẻ đẹp hơn nhưng tổng điểm không tăng
- vùng rìa quanh trụ sinh artifact mới

## 3. Những trick nên làm, nhưng chỉ khi đúng thời điểm

### 3.1. `tower-crop` và `skyline-crop`

Đây không phải hướng xử lý, mà là `trick phân tích`.

Nên dùng khi:

- cần hiểu `B2` đang giúp nền xa bao nhiêu
- cần hiểu local refine có đang chỉ cứu trụ hay không

Không nên dùng như:

- tiêu chí chốt model cuối
- lý do duy nhất để tin rằng model đã tốt hơn

### 3.2. `anchor views`

Giữ một nhóm ảnh mốc:

- vài ảnh test nhìn rõ skyline lỗi
- vài ảnh test nhìn rõ trụ lỗi

Mỗi lần thử trick mới:

- so score tổng
- so trực tiếp đúng các ảnh mốc này

Đây là trick rất rẻ nhưng giúp tránh bị “ảo giác cải thiện”.

### 3.3. `baseline control`

Mỗi trick mới phải có một baseline control cùng điều kiện.

Ví dụ:

- `raw source` vs `prepared source`
- `checkpoint A` vs `checkpoint B`
- `B2 only` vs `B2 + local refine`

Nếu không có control, mọi kết luận đều yếu.

## 4. Những trick không nên mở sớm cho `hcm0031`

### 4.1. `2-model / blend`

Lý do:

- chi phí cao
- khó debug
- chưa cần khi `B2` còn chưa kiểm chứng xong

### 4.2. `ensemble`

Lý do:

- tốn nhiều chạy
- khó biết model nào thật sự tốt
- chưa phù hợp giai đoạn đang sửa lỗi hình học cơ bản

### 4.3. `đổi backbone`

Lý do:

- quá sớm
- làm mất chuẩn đối chiếu trên cùng pipeline hiện có

### 4.4. `segmentation mask` cho tower metric

Lý do:

- hợp lý về mặt nghiên cứu
- nhưng quá nặng so với mục tiêu trước mắt là kiểm chứng `B2`

## 5. Kế hoạch làm trick từ bây giờ đến khi có quyết định

Đây là chuỗi ngắn nhất nên theo:

1. Chạy `Trick 1`:
- dense pilot

2. Nếu `Trick 1` pass:
- chạy `Trick 2`

3. Sau `Trick 2`:
- chạy `Trick 3`

4. Nếu vẫn còn khoảng cải thiện:
- chạy `Trick 4`

5. Chỉ khi nền xa đã tốt hơn thật:
- chạy `Trick 5`

6. Nếu hết chuỗi này mà vẫn không thắng rõ baseline:
- mới cân nhắc `E` hoặc `G`

## 6. Cách dùng file này cho đúng

Mỗi khi nghĩ ra trick mới, phải trả lời 5 câu:

1. Trick này chữa `background xa` hay chữa `trụ`?
2. Nó có đo được trên `round1 public` không?
3. Nó có cơ hội tăng `full-image score` không?
4. Nó có rẻ hơn việc nhảy sang `2-model` hoặc `đổi backbone` không?
5. Nó có đang được thử đúng thứ tự không?

Nếu không trả lời được 5 câu này, trick đó chưa nên ưu tiên.

## 7. Câu chốt để nhớ

- `B2` là bước xử lý quan trọng nhất hiện tại cho `hcm0031`
- `checkpoint selection` là trick rẻ nhất nhưng bắt buộc phải làm
- `local refine` chỉ có giá trị sau khi `B2` đã giúp nền xa
- `crop metric` là công cụ phân tích, không phải tiêu chí chốt cuối
