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


Bài toán yêu cầu thí sinh xây dựng hệ thống AI có khả năng tái dựng cấu trúc 3D ngầm định của trạm BTS từ tập ảnh drone, và sinh ảnh RGB tại các góc nhìn chưa từng được chụp. Đây là hướng tiếp cận hiện đại cho việc xây dựng Digital Twin - bản sao số 3D có độ chính xác cao của hạ tầng viễn thông - phục vụ giám sát, kiểm tra, bảo trì và quy hoạch lắp đặt thiết bị. Mỗi scene gồm 100-300 ảnh RGB kèm thông số camera và pose tương ứng; thí sinh cần sinh ảnh tại 20-50 góc nhìn mục tiêu, đảm bảo đúng về hình học, vị trí thiết bị và chất lượng hình ảnh chân thực.

1. Tổng quan bài toán
Mục tiêu của bài toán là xây dựng mô hình AI có khả năng tái dựng cấu trúc không gian 3D của một scene từ tập ảnh đa góc nhìn và sinh ra ảnh tại các góc nhìn mới chưa từng xuất hiện trong dữ liệu đầu vào.

Dữ liệu có thể được thu thập từ:

Drone bay quanh đối tượng,
Camera cầm tay (hand-held camera).
Đối tượng trong scene có thể là:

Trạm BTS
Công trình hạ tầng
Các đối tượng thực tế khác
Bài toán thuộc các lĩnh vực:

Computer Vision
3D Vision
Neural Rendering
Novel View Synthesis
Digital Twin
2. Cấu trúc dữ liệu
Mỗi scene dữ liệu có cấu trúc như sau:



├── train/
│   ├── images/          : Ảnh training
│   ├── sparse/0/        : Sparse reconstruction từ COLMAP
│   │                       ├── cameras.bin
│   │                       ├── images.bin
│   │                       └── points3D.bin
└── test/
    └── test_poses.csv   : Camera poses cho test images
3. Thông tin dữ liệu
Train images: ~80%
Test images: ~20%
Camera poses và sparse reconstruction đã được dựng sẵn bằng COLMAP và cung cấp cho thí sinh
4. Format test_poses.csv
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
Trong đó:

image_name: tên ảnh đầu ra cần sinh
qw, qx, qy, qz: quaternion rotation theo format COLMAP
tx, ty, tz: camera translation
fx, fy: focal length
cx, cy: principal point
width, height: kích thước ảnh cần sinh
5. Đầu vào bài toán
Đầu vào bao gồm:

tập ảnh train đa góc nhìn
camera intrinsics
camera poses
sparse reconstruction từ COLMAP
danh sách test poses
6. Đầu ra bài toán
Thí sinh cần sinh:

ảnh RGB tương ứng với toàn bộ test poses được cung cấp
Ảnh đầu ra cần:

đúng cấu trúc hình học
đúng vị trí các vật thể
đảm bảo chất lượng hình ảnh chân thực và nhất quán
7. Format submission
Submission là file ZIP chứa toàn bộ ảnh kết quả:

submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
Yêu cầu:

Đúng số lượng và tên scene
Đúng tên file ảnh
Đúng kích thước ảnh
Đúng số lượng ảnh mỗi scene
8. Metrics đánh giá
Kết quả được đánh giá bằng cách so sánh ảnh sinh ra với ảnh ground-truth bằng ba metrics:

8.1 LPIPS
Đánh giá độ tương đồng cảm quan giữa hai ảnh bằng đặc trưng deep learning

Giá trị càng thấp càng tốt.
Tham khảo:

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."
CVPR 2018.
https://arxiv.org/abs/1801.03924
8.2 SSIM
Đánh giá độ tương đồng về cấu trúc hình ảnh

Giá trị càng cao càng tốt.
Tham khảo:

Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
"Image quality assessment: from error visibility to structural similarity."
IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004.
doi: 10.1109/TIP.2003.819861
8.3 PSNR
Đánh giá sai số mức pixel giữa ảnh dự đoán và ground-truth

Giá trị càng cao càng tốt.
Tham khảo:

Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
"Image quality assessment: from error visibility to structural similarity."
IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004.
doi: 10.1109/TIP.2003.819861
Để kết hợp với các metrics khác, giá trị PSNR sẽ được chuẩn hóa về khoảng [0,1] theo công thức:

psnr_norm = torch.clamp(psnr_val / psnr_max, 0.0, 1.0)
Trong đó:

PSNR_max là ngưỡng PSNR tối đa được lựa chọn trước
clamp dùng để giới hạn giá trị trong khoảng từ 0 đến 1
8.4. Công thức tính điểm cuối cùng
S
c
o
r
e
=
0.4
×
(
1
−
L
P
I
P
S
)
+
0.3
×
S
S
I
M
+
0.3
×
P
S
N
R
n
o
r
m
Score=0.4×(1−LPIPS)+0.3×SSIM+0.3×PSNR 
norm
​
 
Điểm trên bảng xếp hạng là điểm trung bình của toàn bộ các scene, nếu thiếu scene hoặc thừa scene so với groundtruth, kết quả sẽ không được tính.

9. Hình thức thi
Dữ liệu và scene hoàn toàn mới được cung cấp cho mỗi vòng thi, cách thức tính điểm sẽ được giữ nguyên.

10. Quy định chống gian lận và đảm bảo tính công bằng
Để đảm bảo cuộc thi đánh giá đúng năng lực xây dựng mô hình AI của thí sinh, Ban Tổ Chức áp dụng các quy định sau:

10.1. Cấm sử dụng dữ liệu ngoài
Thí sinh chỉ được phép sử dụng dữ liệu do Ban Tổ Chức cung cấp trong từng vòng thi.

Nghiêm cấm:

Sử dụng ảnh, video hoặc dữ liệu 3D bên ngoài có chứa cùng đối tượng hoặc cùng scene của bộ dữ liệu thi
Thu thập bổ sung dữ liệu thực địa hoặc từ Internet liên quan trực tiếp đến các scene được cung cấp
Sử dụng bất kỳ nguồn dữ liệu nào nhằm tái tạo hoặc suy luận ground-truth của tập test
10.2. Cấm truy xuất hoặc suy đoán dữ liệu kiểm thử
Nghiêm cấm mọi hành vi nhằm:

Truy cập trái phép vào dữ liệu ground-truth
Khai thác lỗ hổng hệ thống để thu thập thông tin về ảnh kiểm thử
10.3. Yêu cầu khả năng tái lập kết quả
Ban Tổ Chức có quyền yêu cầu các đội đạt thứ hạng cao cung cấp:

Mã nguồn huấn luyện và suy luận
File cấu hình (config)
Danh sách thư viện và phiên bản sử dụng
Checkpoint mô hình
Nhật ký huấn luyện (training logs)
Đội thi phải chứng minh rằng kết quả nộp bài có thể được tái tạo từ pipeline đã công bố.

10.4. Cấm chỉnh sửa thủ công ảnh đầu ra
Toàn bộ ảnh kết quả phải được sinh tự động bởi thuật toán hoặc mô hình AI.

Nghiêm cấm:

Chỉnh sửa thủ công từng ảnh bằng các phần mềm đồ họa
Ghép ảnh, vẽ thêm hoặc xóa vật thể bằng thao tác thủ công
Can thiệp thủ công vào từng test pose
Ban Tổ Chức có quyền yêu cầu chứng minh quy trình sinh ảnh hoàn toàn tự động.

11. Baseline thí sinh có thể tham khảo
https://github.com/graphdeco-inria/gaussian-splatting