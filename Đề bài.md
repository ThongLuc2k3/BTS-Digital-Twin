Đề bài & Quy định
1. Mô tả vòng thi
Đây là vòng thi đầu tiên của bài thi VAR 2026 - Digital Twin cho trạm BTS.

Ở vòng này, ban tổ chức công bố tập public set và private test #1 gồm các scenes khác nhau. Thí sinh xây dựng pipeline và đánh giá trên tập public set. Sau khi công bố tập private test #1, thí sinh sử dụng các ảnh training của mỗi scene để thực hiện sinh ảnh RGB tại các pose mục tiêu được yêu cầu trong file test_pose.csv.

2. Dữ liệu vòng 1
Hạng mục	Thông tin
Số ảnh/scene	150 - 300 ảnh RGB
Số poses mục tiêu/scene	40 - 70
Dung lượng	200 - 300 MB
Cấu trúc dữ liệu giống như đã mô tả trong đề bài chính (xem mục 2.3 Cấu trúc dữ liệu).

3. Yêu cầu submission
Thí sinh nộp một file nén chứa toàn bộ ảnh sinh, theo cấu trúc:

submission_round1.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
Yêu cầu:

Kích thước ảnh: đúng theo width, height trong test_pose.csv
Tên file: theo image_name trong test_pose.csv
Đầy đủ: thiếu ảnh tại bất kỳ pose nào của bất kỳ scene nào sẽ ảnh hưởng đến kết quả
4. Timeline vòng 1
Mốc thời gian	Sự kiện
02/07/2026	Công bố private test #1 - thí sinh tải dữ liệu
30/07/2026	Deadline submission
Thí sinh có thể submit nhiều lần trong thời gian mở. Hệ thống ghi nhận bản submit cuối cùng trước deadline.

5. Lưu ý riêng cho vòng 1
Đây là vòng làm quen với dữ liệu thực tế - hãy kiểm tra kỹ pipeline trên dữ liệu training public trước khi chạy trên private test
Hạ tầng huấn luyện do thí sinh tự chuẩn bị. Hãy ước lượng thời gian chạy để đảm bảo kịp deadline
Cấu hình tham khảo cho mỗi job inference: 1 × RTX A4000 (20 GB VRAM), 4–8 CPU cores, 16–32 GB RAM
Mọi thắc mắc về dữ liệu hoặc submission liên hệ kênh hỗ trợ chính thức của ban tổ chức
Chúc thí sinh thi tốt!

VIETTEL AI RACE 2026 — ĐỀ BTS DIGITAL TWIN

Tài liệu tổng hợp Vòng 1 (Vòng Sơ loại)


1. GIẢI THÍCH ĐỀ TÀI

Tên đề: BTS Digital Twin (Novel View Synthesis)

Tóm tắt 1 câu: Dùng ảnh drone chụp quanh trạm BTS để tái dựng mô hình 3D số hóa, cho phép render ảnh tại bất kỳ góc nhìn nào — kể cả góc chưa từng được chụp.

1.1. Các khái niệm nền

Khái niệmGiải thíchBTS (Base Transceiver Station)Trạm thu phát sóng di động — gồm cột anten, anten phát sóng, tủ thiết bị, dây cáp, giá đỡDigital TwinBản sao số của vật thể thật, cho phép xoay/zoom/đo đạc/theo dõi thay đổi mà không cần tiếp cận trực tiếpNovel View Synthesis (NVS)Kỹ thuật AI cho phép tạo ra ảnh ở góc nhìn mới (chưa từng chụp) từ một tập ảnh gốc đã có3D ReconstructionQuá trình khôi phục hình dạng không gian 3D từ nhiều ảnh 2D

1.2. Quy trình kỹ thuật tổng quát (2 tầng)

Ảnh drone (200-1000 tấm, nhiều góc)
        │
        ▼
[TẦNG 1] Feature Detection → Feature Matching → Camera Pose Estimation
        │   (Trả lời: "Ảnh này chụp từ đâu?")
        │   Công cụ: SIFT/SuperPoint, LightGlue/LoFTR, COLMAP
        ▼
Sparse Point Cloud + Camera Pose
        │
        ▼
[TẦNG 2] Training model 3D (NeRF / 3D Gaussian Splatting)
        │   (Trả lời: "Không gian 3D thật trông như thế nào?")
        ▼
Render Novel View — ảnh 2D tại góc nhìn bất kỳ

Lưu ý quan trọng: 2 tầng có quan hệ tuần tự (Tầng 2 cần input là kết quả của Tầng 1), nhưng việc học và luyện tập công cụ ở mỗi tầng có thể làm song song trên dữ liệu giả lập/công khai, chỉ bước train cuối cùng trên data thật mới cần chờ Tầng 1 xong.


2. YÊU CẦU ĐẦU RA (CHÍNH THỨC TỪ BTC)


Không cần nộp file mô hình 3D trực quan (không cần file xoay 360° tương tác kiểu Polycam/DJI Terra).
Đầu ra bắt buộc: ảnh 2D RGB được render từ các pose (góc nhìn) mục tiêu được yêu cầu trong file test_pose.csv, cho từng scene.



3. QUY ĐỊNH VỀ PIPELINE VÀ MODEL

Quy địnhChi tiếtPipelineKhông bắt buộc end-to-end tự động — được phép xử lý thủ công từng bước, tiền xử lý dữ liệu thô bằng tay trước khi đưa vào modelSố lượng modelKhông giới hạn — được dùng 1 model đơn lẻ HOẶC kết hợp nhiều model (ensemble), fine-tune tự do để tối ưu chất lượng ảnh


4. MÔ TẢ CHI TIẾT VÒNG 1 (chính thức từ trang đề bài)

4.1. Cách vận hành vòng thi


BTC công bố tập public set (để đội tự xây dựng & thử nghiệm pipeline) và tập private test #1 (gồm các scene khác nhau, dùng để nộp bài chính thức).
Sau khi có private test #1: đội dùng ảnh training của mỗi scene để sinh ảnh RGB tại các pose mục tiêu được liệt kê trong test_pose.csv.


4.2. Dữ liệu vòng 1

Hạng mụcThông tinSố ảnh training / scene150 – 300 ảnh RGBSố pose mục tiêu / scene (số ảnh cần sinh ra)40 – 70Dung lượng dữ liệu200 – 300 MB

4.3. Yêu cầu file nộp bài (submission)

Nộp 1 file nén submission_round1.zip, cấu trúc thư mục theo từng scene:

submission_round1.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...

Yêu cầu bắt buộc:


Kích thước ảnh: đúng theo width, height ghi trong test_pose.csv (mỗi scene/pose có thể có kích thước khác nhau — phải kiểm tra kỹ, không dùng 1 kích thước cố định cho tất cả)
Tên file: đặt đúng theo cột image_name trong test_pose.csv
Đầy đủ: thiếu ảnh ở bất kỳ pose nào của bất kỳ scene nào đều ảnh hưởng điểm số — cần kiểm tra đủ số lượng trước khi nộp


4.4. Timeline vòng 1

Mốc thời gianSự kiện02/07/2026Công bố private test #1 — thí sinh tải dữ liệu30/07/2026Deadline submission


Được nộp nhiều lần trong thời gian mở vòng thi.
Hệ thống chỉ ghi nhận bản nộp cuối cùng trước deadline (không phải điểm cao nhất trong các lần nộp).


4.5. Giới hạn kỹ thuật khi nộp bài

MụcGiá trịLoại bài nộpTệp ZIPHạ tầng chấm điểmGPUGiới hạn số lần nộp5 lần/ngàyThời gian chờ giữa các lần chấm600 giây (10 phút)

4.6. Hạ tầng tính toán


Hạ tầng huấn luyện (training) do thí sinh tự chuẩn bị — vòng Sơ loại chưa được BTC cấp GPU, cần tự ước lượng thời gian chạy để kịp deadline 30/7.
Cấu hình tham khảo cho mỗi job inference (khi BTC chấm bài): 1× RTX A4000 (20 GB VRAM), 4–8 CPU cores, 16–32 GB RAM. → Đội nên thiết kế model/pipeline chạy vừa trong cấu hình tương đương này để đảm bảo tương thích khi BTC inference lại.


4.7. Lưu ý riêng từ BTC


Đây là vòng làm quen với dữ liệu thực tế — cần kiểm tra kỹ pipeline trên tập public set trước khi chạy chính thức trên private test.
Mọi thắc mắc về dữ liệu/submission → liên hệ kênh hỗ trợ chính thức của BTC (không hỏi qua kênh không chính thức).



5. MỐC THỜI GIAN TOÀN CUỘC THI

MốcThời gianCông bố private test #1 (Vòng 1)02/07/2026Livestream giải thích thể lệ "Sẵn sàng nhập cuộc và bứt tốc"12/07/2026Deadline nộp bài Vòng 1 (Sơ loại)30/07/2026Kết quả chọn đội vào Vòng 2 (Sơ khảo)Sau 30/7, chọn 24 đội tốt nhấtVòng 2 — Sơ khảo (trực tiếp, Hà Nội)17/08 – 19/08/2026Vòng 3 — Chung khảo (trực tiếp)09/09 – 10/09/2026Lễ trao giải11/09/2026


6. NHỮNG ĐIỂM CẦN TỰ XÁC NHẬN THÊM


Chưa rõ công khai việc có được dùng AI hỗ trợ viết code (Copilot, Claude Code...) hay không — nên hỏi qua kênh hỗ trợ chính thức của BTC nếu cần chắc chắn.
Chưa rõ dữ liệu training mỗi scene có kèm sẵn camera pose/metadata hay đội phải tự chạy SfM (COLMAP) để tính pose — cần kiểm tra ngay khi tải public set (mục 2.3 "Cấu trúc dữ liệu" trong đề bài chính).
Cách tính PSNR/SSIM/LPIPS cụ thể (trên toàn ảnh hay có vùng loại trừ như nền/sky) — nếu đề bài chính không nêu rõ, nên hỏi BTC.



7. CHIẾN LƯỢC KỸ THUẬT ĐỀ XUẤT CHO ĐỘI

Tầng 1 — Feature Matching (phụ trách: thành viên 3D/Graphics)


Chính: LightGlue
So sánh thêm: DALGlue (chuyên UAV), LoFTR
Công cụ pose: COLMAP


Tầng 2 — Reconstruction/NVS (phụ trách: CV core)


Chính: DroneSplat (3DGS cải tiến sẵn cho ảnh drone in-the-wild)
So sánh/ensemble thêm: 3D Gaussian Splatting gốc, Nerfacto, LOBE-GS (nếu cảnh lớn)


Chiến lược ensemble (vì đề cho phép kết hợp nhiều model)


Train song song 2-3 model NVS
Với mỗi góc test, chọn hoặc blend kết quả từ model cho điểm PSNR/SSIM cao nhất
Xử lý riêng đặc thù trạm BTS: khung thép mảnh, dây cáp, ánh sáng ngoài trời thay đổi



8. PHÂN CÔNG NHIỆM VỤ 3 THÀNH VIÊN

Vai tròNhiệm vụ chínhCV core (em)Train nhiều model NVS (3DGS, DroneSplat, Nerfacto), render ảnh góc test, tối ưu/cải tiến3D/GraphicsChạy Feature Matching + COLMAP, xử lý/kiểm tra camera pose, chuẩn hóa dữ liệu đầu vàoIT tổng quátViết script tính điểm (PSNR/SSIM/LPIPS), script ensemble/blend kết quả, đóng gói nộp bài đúng format sample_submission


Tài liệu tổng hợp dựa trên thông báo chính thức từ BTC (cập nhật đến 04/07/2026) và thông tin công khai từ trang competition.viettel.vn. Một số chi tiết kỹ thuật (dataset, giới hạn nộp bài) sẽ được cập nhật khi BTC công bố thêm.
Chi tiết vòng thi

Loại bài nộp
Tệp ZIP
Hạ tầng chấm
GPU
Giới hạn nộp bài
5 lần/ngày
Thời gian chờ
600 giây