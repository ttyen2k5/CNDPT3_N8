ĐỀ TÀI 8.3. Hệ thống nhận diện hành động và tương tác trong VR bằng AI (Action Recognition)
Mô tả:
AI phân tích chuyển động cơ thể (từ camera hoặc sensor) để nhận diện hành động → điều khiển tương tác trong VR (ví dụ: vẫy tay, chạy, nhảy).

Công nghệ:

Pose Estimation (MediaPipe, OpenPose)
Action Recognition (LSTM, 3D CNN)
VR Interaction System
Điểm mới:

Tăng tính “natural interaction” trong VR
Ứng dụng mạnh trong game, fitness, training
Mở rộng:

Nhận diện hành vi bất thường (an ninh, giám sát)
Kết hợp AI dự đoán hành động tiếp theo


LUỒNG HOẠT ĐỘNG CỦA HỆ THỐNG:
Camera → Pose Estimation → Nhận diện hành động → Điều khiển → Hiển thị

1: Code webcam + hiển thị video
Code phần:
Mở webcam bằng Python
Hiển thị video trên màn hình
Đảm bảo camera chạy ổn định
Sản phẩm:
Chạy webcam được
Video hiển thị mượt
Có nút thoát (ESC)

2: Code Pose Estimation
Code phần:
Dùng MediaPipe để nhận diện khung xương
Hiển thị khung xương trên người
Lấy tọa độ tay, chân, đầu (keypoints)
Sản phẩm:
Khi đứng trước webcam → hiện khung xương
In ra tọa độ tay hoặc đầu
3: Code nhận diện hành động
Code phần:
Nhận dữ liệu từ người 2
Viết điều kiện nhận diện hành động:
giơ tay (Raise Hand)
vẫy tay (Wave)
chạy tại chỗ (Running)
đi bộ tại chỗ (Walking)
nhảy (Jump)
cúi người (Bend)
Sản phẩm:
Khi giơ tay → hiện chữ: "Raise Hand"
Khi vẫy tay → hiện chữ: "Wave"
Khi chạy tại chỗ → hiện chữ: "Running"
Khi đi bộ tại chỗ → hiện chữ: "Walking"
Khi bật nhảy → hiện chữ: "Jump"
Khi cúi người → hiện chữ: "Bend"

4: Code phần điều khiển (VR / game demo)
Code phần:
Khi AI nhận diện hành động → thực hiện hành động
Ví dụ:
giơ tay → nhân vật giơ tay
vẫy tay → mở menu
chạy → nhân vật di chuyển nhanh
đi bộ → nhân vật di chuyển chậm
nhảy → nhân vật nhảy lên
cúi người → nhân vật cúi
Nếu không làm VR thật thì có thể làm:
game nhỏ bằng Python
Bước 1: Tạo nhân vật
Bước 2: Nhận hành động từ AI
Bước 3: Làm từng hành động
hoặc nhân vật di chuyển trên màn hình
Sản phẩm:
Có hành động xảy ra khi người dùng cử động

5: Code giao diện + hiển thị kết quả
Code phần:
Hiển thị tên hành động trên màn hình
Hiển thị FPS
Hiển thị trạng thái hệ thống
Ví dụ:
Action: Running
FPS: 28
Status: Tracking…

