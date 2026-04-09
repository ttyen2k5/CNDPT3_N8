# BÁO CÁO TỔNG KẾT ĐỀ TÀI
**Tên đề tài:** Hệ thống nhận diện hành động và tương tác trong môi trường ảo (VR/Game) bằng Trí tuệ nhân tạo (Action Recognition)
**Mã đề tài:** 8.3

---

## MỤC LỤC
1. [Chương 1: Tổng quan đề tài](#chuong-1-tong-quan-de-tai)
2. [Chương 2: Cơ sở lý thuyết và Công nghệ sử dụng](#chuong-2-co-so-ly-thuyet-va-cong-nghe-su-dung)
3. [Chương 3: Phân tích và Thiết kế hệ thống](#chuong-3-phan-tich-va-thiet-ke-he-thong)
4. [Chương 4: Triển khai và Tích hợp Game](#chuong-4-trien-khai-va-tich-hop-game)
5. [Chương 5: Kết luận và Hướng phát triển](#chuong-5-ket-luan-va-huong-phat-trien)

---

## CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI

### 1.1. Đặt vấn đề
Trong những năm gần đây, công nghệ Thực tế ảo (VR) và các trò chơi tương tác chuyển động (Motion-sensing games) đang ngày càng phát triển mạnh mẽ. Tuy nhiên, các hệ thống truyền thống thường yêu cầu người dùng phải đeo các thiết bị cảm biến phức tạp (như tay cầm VR, Kinect, thiết bị đeo trên người) gây ra sự bất tiện và tốn kém chi phí. 
Với sự tiến bộ của Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision), việc nhận diện hành động của con người thông qua một camera (webcam) thông thường đã trở nên khả thi. Điều này mở ra cơ hội xây dựng các hệ thống tương tác tự nhiên (Natural Interaction) mà không cần thiết bị phần cứng chuyên dụng.

### 1.2. Mục tiêu đề tài
Mục tiêu của đề tài là xây dựng một hệ thống AI hoàn chỉnh có khả năng:
- **Nhận diện khung xương (Pose Estimation):** Trích xuất các điểm khớp nối trên cơ thể người dùng theo thời gian thực thông qua webcam.
- **Phân loại hành động (Action Recognition):** Sử dụng mô hình Học máy (Machine Learning) để phân loại các hành động cụ thể (Nhảy lên - Jump, Cúi người - Bend).
- **Tương tác thời gian thực (Real-time Interaction):** Áp dụng kết quả nhận diện để điều khiển một nhân vật trong trò chơi (Dodge Game - Né chướng ngại vật) với độ trễ thấp nhất.

### 1.3. Phạm vi đề tài
- Hệ thống tập trung nhận diện 2 hành động chính mang tính thể chất: **Jump** (Nhảy) và **Bend** (Cúi người).
- Trạng thái mặc định khi không có hành động rõ ràng là **Idle** (Đứng yên).
- Môi trường thử nghiệm: Trò chơi 2D né chướng ngại vật được lập trình trực tiếp bằng OpenCV.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG

### 2.1. Bài toán Pose Estimation và MediaPipe
**Pose Estimation** là bài toán xác định vị trí và hướng của các bộ phận trên cơ thể người trong không gian 2D hoặc 3D từ hình ảnh hoặc video.
Trong đề tài này, thư viện **MediaPipe** (của Google) được sử dụng thông qua giao diện **Tasks API (PoseLandmarker)**. MediaPipe cung cấp mô hình nhẹ (Lite model) có khả năng chạy mượt mà trên CPU, trích xuất chính xác **33 điểm mốc (landmarks)** trên cơ thể người với các tọa độ không gian (x, y, z) và độ tin cậy (visibility).

### 2.2. Bài toán Action Recognition và Random Forest
Thay vì sử dụng các luật cứng (Rule-based/Heuristics) dễ bị sai số do sự khác biệt về chiều cao và góc máy, đề tài tiếp cận bằng phương pháp **Data-driven (Học máy)**.
- **Thuật toán sử dụng:** Random Forest Classifier (thuộc thư viện Scikit-Learn).
- **Lý do lựa chọn:** Random Forest hoạt động rất hiệu quả với dữ liệu dạng bảng (tabular data), có khả năng chống overfitting tốt, không yêu cầu chuẩn hóa dữ liệu quá khắt khe và tốc độ suy luận (inference) cực kỳ nhanh, hoàn toàn đáp ứng được yêu cầu real-time của game.

### 2.3. Các công cụ và thư viện khác
- **OpenCV (cv2):** Xử lý luồng video từ webcam, lật ảnh (mirror effect), và vẽ đồ họa 2D cho trò chơi.
- **NumPy:** Tính toán ma trận, xử lý và biến đổi các vector đặc trưng (Feature extraction).
- **Joblib:** Lưu trữ và tải mô hình Machine Learning đã được huấn luyện.
- **Jupyter Notebook:** Môi trường để phân tích dữ liệu, thử nghiệm và huấn luyện mô hình.

---

## CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

### 3.1. Kiến trúc tổng thể
Hệ thống được chia làm 2 giai đoạn (Phases) hoạt động độc lập nhưng liên kết chặt chẽ với nhau:
1. **Giai đoạn Huấn luyện (Offline Training):** Thu thập video $\rightarrow$ Trích xuất đặc trưng $\rightarrow$ Huấn luyện mô hình $\rightarrow$ Xuất file `model.pkl`.
2. **Giai đoạn Suy luận (Real-time Inference):** Đọc Webcam $\rightarrow$ Trích xuất đặc trưng $\rightarrow$ Mô hình dự đoán $\rightarrow$ Lọc nhiễu $\rightarrow$ Điều khiển Game.

### 3.2. Quy trình trích xuất đặc trưng (Feature Extraction)
Để mô hình Random Forest có thể hiểu được chuỗi chuyển động, dữ liệu thô từ MediaPipe cần được chuyển đổi:
- Mỗi khung hình (frame) chứa 33 điểm, mỗi điểm có 3 tọa độ (x, y, z). Tổng cộng: $33 \times 3 = 99$ chiều.
- Đối với một video (hoặc một cửa sổ thời gian n frames), hệ thống sẽ tính toán các đại lượng thống kê trên trục thời gian cho từng tọa độ: **Mean (Trung bình), Standard Deviation (Độ lệch chuẩn), Min (Nhỏ nhất), Max (Lớn nhất)**.
- Kết quả: Mỗi chuỗi chuyển động được biểu diễn bằng một vector cố định có kích thước $99 \times 4 = 396$ chiều.
- Cuối cùng, vector này được chuẩn hóa (Z-score normalization) để tăng tính ổn định.

### 3.3. Xây dựng mô hình (Training Pipeline)
Quá trình huấn luyện được thực hiện trong tệp `train.ipynb`:
1. Quét toàn bộ video trong thư mục `data/` (`jump_*.mp4`, `bend_*.mp4`).
2. Gọi hàm trích xuất đặc trưng để biến mỗi video thành 1 vector 396 chiều.
3. Gán nhãn: `Jump = 1`, `Bend = 0`.
4. Chia tập dữ liệu tự động theo tỷ lệ **80% Training** và **20% Testing** (đảm bảo cân bằng lớp).
5. Huấn luyện mô hình Random Forest với 300 cây quyết định (n_estimators=300).
6. Đánh giá mô hình qua các chỉ số: Accuracy, Confusion Matrix, Classification Report.
7. Đóng gói mô hình cùng với metadata (nhãn, kích thước feature) vào tệp `model.pkl`.

---

## CHƯƠNG 4: TRIỂN KHAI VÀ TÍCH HỢP GAME

### 4.1. Suy luận thời gian thực (Real-time Inference)
Trong tệp `main.py` và module `action_recognizer.py`, hệ thống xử lý luồng video trực tiếp:
- **Cửa sổ trượt (Sliding Window):** Hệ thống lưu trữ các frame gần nhất vào một hàng đợi (deque) có kích thước `MODEL_FEATURE_WINDOW = 5`.
- **Dự đoán:** Khi hàng đợi đủ dữ liệu, hệ thống tính toán vector 396 chiều và đưa vào mô hình để lấy nhãn dự đoán cùng với xác suất (Confidence Score).
- **Cơ chế Lọc nhiễu (Smoothing & Thresholding):**
  - Nếu Confidence Score $<$ `MODEL_CONFIDENCE_THRESHOLD` (0.70), hệ thống coi hành động đó không rõ ràng và gán trạng thái là **Idle**.
  - Sử dụng cơ chế **Majority Vote** (Bầu chọn theo số đông) trên các dự đoán gần nhất để tránh hiện tượng nhân vật bị giật lag (flickering) khi chuyển đổi trạng thái.

### 4.2. Thiết kế Game Tương tác (Dodge Game)
Trò chơi được xây dựng trong module `game_controller.py` với các quy tắc:
- **Màn hình:** Chia làm 2 phần, bên trái là luồng camera thực tế, bên phải là giao diện Game 2D.
- **Nhân vật (Stick figure):** Đứng ở vị trí cố định bên trái màn hình game.
- **Chướng ngại vật:** Di chuyển từ phải sang trái với tốc độ tăng dần theo điểm số. Có 2 loại:
  1. **Gai nhọn dưới đất (Low obstacle):** Yêu cầu người dùng phải thực hiện hành động **Jump** để nhân vật nhảy qua.
  2. **Thanh ngang trên cao (High obstacle):** Yêu cầu người dùng phải thực hiện hành động **Bend** để nhân vật cúi người né tránh.
- **Luật chơi:** Người chơi có 3 mạng (Lives). Mỗi lần va chạm sẽ mất 1 mạng và màn hình nhấp nháy đỏ. Khi hết mạng, trò chơi kết thúc (Game Over) và có thể nhấn phím `R` để chơi lại.

---

## CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 5.1. Kết quả đạt được
- Xây dựng thành công một hệ thống Action Recognition hoàn chỉnh từ khâu thu thập dữ liệu, huấn luyện AI đến suy luận thời gian thực.
- Mô hình Random Forest kết hợp với các đặc trưng thống kê (Mean, Std, Min, Max) cho độ chính xác cao và tốc độ xử lý cực nhanh, không gây độ trễ cho game.
- Trò chơi hoạt động mượt mà, logic vật lý và va chạm (collision detection) phản hồi chính xác với hành động thực tế của người dùng.

### 5.2. Hạn chế
- Mô hình hiện tại sử dụng đặc trưng thống kê tổng hợp (Aggregated features) nên có thể mất đi một số thông tin về thứ tự thời gian (temporal dynamics) của chuỗi chuyển động phức tạp.
- Phụ thuộc vào chất lượng ánh sáng và góc đặt camera (cần thấy rõ toàn thân hoặc nửa thân trên).

### 5.3. Hướng phát triển
- **Nâng cấp mô hình AI:** Thay thế Random Forest bằng các mạng nơ-ron chuyên xử lý chuỗi thời gian như **LSTM (Long Short-Term Memory)** hoặc **ST-GCN (Spatial Temporal Graph Convolutional Networks)** để nhận diện các hành động phức tạp hơn (ví dụ: đấm, đá, vẫy tay theo nhịp).
- **Mở rộng dữ liệu:** Tích hợp các kỹ thuật Data Augmentation (xoay, lật, thêm nhiễu vào tọa độ keypoints) để mô hình bền vững (robust) hơn với nhiều góc máy khác nhau.
- **Tích hợp Engine chuyên nghiệp:** Chuyển đổi giao diện game 2D OpenCV sang các Game Engine hiện đại như Unity hoặc Unreal Engine để tạo ra môi trường VR/AR chân thực và sống động hơn.
