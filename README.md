# Hệ thống nhận diện hành động và tương tác trong VR bằng AI

**ĐỀ TÀI 8.3 - Action Recognition for VR Interaction**

---

## 1. Mô tả tổng quan

AI phân tích chuyển động cơ thể người dùng thông qua webcam, nhận diện khung xương (Pose Estimation), từ đó xác định hành động đang thực hiện (Action Recognition), và điều khiển nhân vật trong môi trường VR/game demo tương ứng.

### Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Pose Estimation | MediaPipe PoseLandmarker (Tasks API) | Nhận diện 33 keypoints khung xương |
| AI Model | TensorFlow Lite (pose_landmarker_lite) | Chạy inference nhận diện pose |
| Action Recognition | Rule-based heuristics | Phân tích keypoints → xác định hành động |
| Camera & Display | OpenCV (cv2) | Thu hình webcam, vẽ khung xương, hiển thị |
| Game Demo | OpenCV Drawing | Nhân vật stick figure phản hồi hành động |
| Ngôn ngữ | Python 3.10+ | |

---

## 2. Kiến trúc hệ thống (System Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main.py (Entry Point)                      │
│                                                                     │
│  Khởi tạo modules → Vòng lặp chính → Ghép frame → Hiển thị → ESC  │
└───────┬──────────────┬───────────────┬──────────────┬───────────────┘
        │              │               │              │
        ▼              ▼               ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐
  │ camera   │  │   pose     │  │  action    │  │  game    │
  │  .py     │  │ estimator  │  │ recognizer │  │controller│
  │          │  │   .py      │  │   .py      │  │   .py    │
  │ Camera   │  │ Pose       │  │ Action     │  │ Game     │
  │ class    │  │ Estimator  │  │ Recognizer │  │Controller│
  └──────────┘  └────────────┘  └────────────┘  └──────────┘
        │              │               │              │
        │              │               │              │
        ▼              ▼               ▼              ▼
  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐
  │ OpenCV   │  │ MediaPipe  │  │ Deque      │  │ OpenCV   │
  │VideoCapt.│  │ Tasks API  │  │ History    │  │ Drawing  │
  └──────────┘  └────────────┘  └────────────┘  └──────────┘
                                                      │
                                                      ▼
                                                ┌──────────┐
                                                │ui_overlay│
                                                │   .py    │
                                                │ UIOverlay│
                                                └──────────┘
```

---

## 3. Luồng hoạt động tổng thể (Main Flow)

```
                    ┌──────────────┐
                    │  Khởi động   │
                    │  main.py     │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Khởi tạo 5 modules:    │
              │  Camera                │
              │  PoseEstimator         │
              │  ActionRecognizer      │
              │  GameController        │
              │  UIOverlay             │
              └────────────┬───────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Tải model AI (lần   │
                │ đầu: download từ    │
                │ Google Storage)     │
                └──────────┬──────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Mở webcam     │
                  │  (1280 x 720)  │
                  └───────┬────────┘
                          │
          ┌───────────────┘
          │
          ▼
┌─── VÒNG LẶP CHÍNH (mỗi frame ~33ms) ──────────────────────┐
│                                                              │
│  ┌──────────────────┐                                        │
│  │ 1. Đọc frame     │  Camera.read()                        │
│  │    từ webcam      │  → BGR frame (1280x720)               │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 2. Lật gương     │  cv2.flip(frame, 1)                   │
│  │    (mirror)       │  → người dùng thấy như soi gương      │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 3. Pose          │  PoseEstimator.process(frame)          │
│  │    Estimation     │  → keypoints dict + annotated frame   │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 4. Nhận diện     │  ActionRecognizer.recognize(keypoints) │
│  │    hành động      │  → "Wave"/"Raise Hand"/"Bend"/...     │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 5. Cập nhật      │  GameController.update(action)         │
│  │    game demo      │  GameController.draw() → game panel   │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 6. Vẽ UI         │  UIOverlay.draw(frame, action, fps)    │
│  │    overlay        │  → frame với Action/FPS/Status         │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 7. Ghép &        │  np.hstack([camera_frame, game_panel]) │
│  │    hiển thị       │  cv2.imshow() → màn hình               │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                        │
│  │ 8. Kiểm tra      │  ESC → thoát                           │
│  │    phím nhấn      │  D   → bật/tắt debug overlay          │
│  └────────┬─────────┘                                        │
│           │                                                  │
│           └──────────── Quay lại bước 1 ◄───────────         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼ (ESC hoặc Ctrl+C)
                ┌────────────────────┐
                │ Giải phóng:        │
                │  camera.release()  │
                │  pose.release()    │
                │  destroyAllWindows │
                └────────────────────┘
```

---

## 4. Flow chi tiết từng module

### 4.1 Phần 1: Camera (`modules/camera.py`)

**Class `Camera`** - Quản lý webcam

```
Camera.__init__(camera_id=0)
    │
    ▼
Camera.open()
    │
    ├── cv2.VideoCapture(0)        ← mở webcam
    ├── set FRAME_WIDTH = 1280     ← độ phân giải
    ├── set FRAME_HEIGHT = 720
    │
    ├── Thành công? ──► return self
    └── Thất bại?   ──► raise RuntimeError
    
Camera.read()                      ← gọi mỗi frame
    │
    └── return (success: bool, frame: numpy.ndarray BGR)

Camera.release()                   ← giải phóng cuối chương trình
    │
    └── cap.release()
```

**Sản phẩm:** Video webcam chạy mượt, có nút thoát ESC.

---

### 4.2 Phần 2: Pose Estimation (`modules/pose_estimator.py`)

**Class `PoseEstimator`** - Nhận diện 33 keypoints khung xương

```
PoseEstimator.__init__()
    │
    ├── _ensure_model()
    │       │
    │       ├── File .task đã có?  ──► bỏ qua
    │       └── Chưa có? ──► Download từ Google Storage (~4MB)
    │                         pose_landmarker_lite.task
    │
    ├── Đọc model vào RAM (model_asset_buffer)
    │   ← Tránh lỗi Unicode path trên Windows
    │
    └── Tạo PoseLandmarker
            mode = VIDEO
            num_poses = 1
            detection_confidence = 0.5
            tracking_confidence = 0.5
```

**Flow xử lý mỗi frame:**

```
PoseEstimator.process(frame)
    │
    ├── 1. Chuyển BGR → RGB
    │       cv2.cvtColor(frame, COLOR_BGR2RGB)
    │
    ├── 2. Tạo MediaPipe Image
    │       mp.Image(format=SRGB, data=rgb)
    │
    ├── 3. Tính timestamp
    │       frame_idx * 33ms (~30 FPS)
    │
    ├── 4. Chạy inference AI
    │       landmarker.detect_for_video(image, timestamp)
    │       ← TensorFlow Lite XNNPACK delegate (CPU)
    │
    ├── 5. Có phát hiện người?
    │       │
    │       ├── CÓ:
    │       │   ├── Trích xuất 14 keypoints chính:
    │       │   │     nose, left/right_shoulder, left/right_elbow,
    │       │   │     left/right_wrist, left/right_hip,
    │       │   │     left/right_knee, left/right_ankle
    │       │   │
    │       │   ├── Format: {name: (x, y, z, visibility)}
    │       │   │     x, y: tọa độ chuẩn hóa [0.0 → 1.0]
    │       │   │     y=0 là top, y=1 là bottom
    │       │   │
    │       │   └── Vẽ khung xương lên frame
    │       │         33 điểm (chấm xanh lá, r=5)
    │       │         + đường nối (33 connections)
    │       │         Chỉ vẽ nếu visibility > 0.5
    │       │
    │       └── KHÔNG:
    │           └── keypoints = {} (rỗng)
    │
    └── return (keypoints_dict, annotated_frame)
```

**14 Keypoints được trích xuất:**

```
        [0] nose
          |
   [11]---+---[12]         left_shoulder ── right_shoulder
    |             |
  [13]          [14]        left_elbow ── right_elbow
    |             |
  [15]          [16]        left_wrist ── right_wrist
          |
   [23]---+---[24]         left_hip ── right_hip
    |             |
  [25]          [26]        left_knee ── right_knee
    |             |
  [27]          [28]        left_ankle ── right_ankle
```

---

### 4.3 Phần 3: Action Recognition (`modules/action_recognizer.py`)

**Class `ActionRecognizer`** - Nhận diện 6 hành động từ keypoints

**Flow nhận diện chính:**

```
ActionRecognizer.recognize(keypoints)
    │
    ├── keypoints rỗng? ──► return "No Person"
    │
    ├── Kiểm tra 6 hành động:
    │       wave, raise_hand, running, walking, jump, bend
    │
    ├── Xác định raw action theo ƯU TIÊN:
    │       Wave > Raise Hand > Running > Walking > Jump > Bend > Idle
    │
    └── Hysteresis Filter (chống nhấp nháy):
            │
            ├── raw action GIỐNG action trước?
            │     └── Áp dụng ngay
            │
            └── raw action KHÁC action trước?
                  └── Phải lặp lại 2 frame liên tiếp
                      mới chuyển sang action mới
```

---

#### 4.3.1 Thuật toán: Raise Hand (giơ tay)

```
_is_raise_hand(keypoints)
    │
    ├── Lấy tọa độ y: left_wrist, left_shoulder
    ├── Lấy tọa độ y: right_wrist, right_shoulder
    │
    ├── Điều kiện (tay trái HOẶC tay phải):
    │     wrist.y < shoulder.y - 0.10
    │     ← cổ tay cao hơn vai ít nhất 10% chiều cao frame (~72px)
    │     ← y nhỏ hơn = vị trí cao hơn (0=top, 1=bottom)
    │
    ├── TRUE  → đang giơ tay
    └── FALSE → không giơ tay

    Ví dụ: shoulder.y = 0.40, wrist.y = 0.25
           0.25 < 0.40 - 0.10 = 0.30 → TRUE ✓
```

---

#### 4.3.2 Thuật toán: Wave (vẫy tay)

```
_is_wave(keypoints)
    │
    ├── Bước 1: Kiểm tra tay có giơ lên không
    │     └── _is_raise_hand() == False? ──► clear history, return False
    │
    ├── Bước 2: Sticky Check (giữ Wave liên tục)
    │     │
    │     ├── _wave_sticky > 0?  (đang trong thời gian giữ)
    │     │     ├── Giảm _wave_sticky
    │     │     ├── Vẫn thu thập wrist_x vào history
    │     │     ├── Kiểm tra lại pattern → nếu vẫn vẫy → reset timer = 15
    │     │     └── return True (giữ Wave)
    │     │
    │     └── _wave_sticky == 0? → tiếp tục bước 3
    │
    ├── Bước 3: Thu thập vị trí cổ tay
    │     ├── Chọn tay nào giơ cao hơn (y nhỏ hơn)
    │     └── Thêm wrist_x vào history (tối đa 25 frame)
    │
    ├── Bước 4: Chưa đủ 8 frame? ──► return False
    │
    └── Bước 5: _check_wave_pattern()
            │
            ├── Tính biên độ x_range = max(history) - min(history)
            │     └── x_range < 0.018 (23px)? ──► False (vẫy quá nhỏ)
            │
            ├── Tính velocity mỗi frame:
            │     vel[i] = history[i] - history[i-1]
            │
            ├── Đếm số lần đổi chiều (sign changes):
            │     vel trước > +0.002 VÀ vel sau < -0.002 → +1
            │     vel trước < -0.002 VÀ vel sau > +0.002 → +1
            │     (dead_zone = 0.002 lọc nhiễu micro-movement)
            │
            ├── sign_changes >= 2? ──► TRUE → set sticky = 15 frame
            └── Ngược lại       ──► FALSE

    Ví dụ chuỗi wrist_x khi vẫy:
    [0.45, 0.47, 0.50, 0.48, 0.44, 0.42, 0.44, 0.47, 0.50]
     ──────►  ◄──────────  ──────►
     vel: +    +    -    -    -    +    +    +
                  ↑ đổi chiều       ↑ đổi chiều → 2 lần ✓
```

---

#### 4.3.3 Thuật toán: Bend (cúi người)

```
_is_bend(keypoints)
    │
    ├── Lấy tọa độ: nose, left/right_shoulder, left/right_hip
    │
    ├── Tính trung bình:
    │     shoulder_y = avg(left_shoulder.y, right_shoulder.y)
    │     hip_y      = avg(left_hip.y, right_hip.y)
    │
    ├── Khoảng cách vai-hông:
    │     sh_to_hip = hip_y - shoulder_y
    │
    ├── sh_to_hip < 0.03? ──► TRUE (thân gập hoàn toàn)
    │
    ├── Tính tỷ lệ chiều cao:
    │     nose_to_hip  = hip_y - nose.y     ← mũi cách hông bao xa
    │     height_ratio = nose_to_hip / sh_to_hip
    │
    │     ┌─────────────────────────────────────────────────┐
    │     │ Đứng thẳng: ratio ≈ 1.6  (mũi cao hơn vai nhiều)│
    │     │ Cúi nhẹ:    ratio ≈ 1.3  (mũi bắt đầu hạ)     │
    │     │ Cúi vừa:    ratio ≈ 1.0  (mũi ngang vai)       │
    │     │ Cúi sâu:    ratio ≈ 0.7  (mũi dưới vai)        │
    │     └─────────────────────────────────────────────────┘
    │
    └── height_ratio < 1.35? ──► TRUE (đang cúi)
                             ──► FALSE (đứng thẳng)

    Ví dụ cúi nhẹ:
      nose.y = 0.42, shoulder_y = 0.43, hip_y = 0.65
      nose_to_hip  = 0.65 - 0.42 = 0.23
      sh_to_hip    = 0.65 - 0.43 = 0.22
      height_ratio = 0.23 / 0.22 = 1.05 < 1.35 → TRUE ✓
```

---

#### 4.3.4 Thuật toán: Running (chạy tại chỗ)

```
_is_running(keypoints)
    │
    ├── Lấy tọa độ: left_ankle.y, right_ankle.y
    ├── Thêm vào ankle_history (tối đa 30 frame)
    ├── Chưa đủ 12 frame? ──► return False
    │
    ├── KIỂM TRA 1: Biên độ dao động đủ lớn
    │     l_range = max(left_ys)  - min(left_ys)
    │     r_range = max(right_ys) - min(right_ys)
    │
    │     CẢ HAI < 0.055 (40px)? ──► False (chỉ là đi bộ/đứng yên)
    │
    ├── KIỂM TRA 2: Tốc độ dao động đủ nhanh
    │     l_speed = trung bình |delta_y| mỗi frame (trái)
    │     r_speed = trung bình |delta_y| mỗi frame (phải)
    │
    │     speed < 0.008? ──► False (chuyển động quá chậm)
    │
    └── KIỂM TRA 3: Luân phiên trái-phải
          │
          ├── diff[i] = left_ankle_y[i] - right_ankle_y[i]
          │     ← dương: chân trái thấp hơn (cao hơn trên camera)
          │     ← âm:    chân phải thấp hơn
          │
          ├── Đếm sign_changes (ngưỡng 0.015):
          │     diff đổi dấu từ +0.015 sang -0.015 → +1
          │     diff đổi dấu từ -0.015 sang +0.015 → +1
          │
          └── sign_changes >= 3? ──► TRUE (đang chạy tại chỗ)
                                 ──► FALSE

    Phân biệt đi bộ vs chạy:
    ┌──────────┬────────────┬──────────┬──────────────┐
    │          │ Biên độ    │ Tốc độ   │ Luân phiên   │
    ├──────────┼────────────┼──────────┼──────────────┤
    │ Đứng yên │ < 0.01     │ < 0.002  │ 0            │
    │ Đi bộ    │ 0.02-0.04  │ 0.003-06 │ 1-2          │
    │ Chạy     │ > 0.06     │ > 0.01   │ 3+       ✓   │
    └──────────┴────────────┴──────────┴──────────────┘
```

---

#### 4.3.5 Thuật toán: Walking (đi bộ tại chỗ)

Tương tự Running nhưng biên độ/tốc độ thấp hơn:
- Biên độ mắt cá >= 2% nhưng <= 5.5% (trên 5.5% = Running)
- Luân phiên trái-phải >= 2 lần

#### 4.3.6 Thuật toán: Jump (nhảy)

Hông nâng lên đủ cao so với baseline (nửa đầu) của hip_history:
- `baseline - hip_y_hiện_tại > 4%` → đang nhảy

---

### 4.4 Phần 4: Game Controller (`modules/game_controller.py`)

**Class `GameController`** - Nhân vật stick figure phản hồi hành động

```
GameController.update(action)
    │
    ├── action == "Jump"?
    │     └── Chưa đang nhảy → kích hoạt nhảy
    │
    ├── action == "Running"?
    │     └── Di chuyển nhanh sang phải
    │
    ├── action == "Walking"?
    │     └── Di chuyển chậm sang phải
    │
    ├── action == "Wave"?
    │     └── menu_alpha tăng dần → hiện menu overlay
    │         (Resume Game, Settings, Inventory, Exit)
    │
    └── Vật lý nhảy (mỗi frame):
          y += vel_y
          vel_y += gravity (1)
          Chạm đất? → dừng nhảy

GameController.draw() → numpy.ndarray (400 x 720)
    │
    ├── Vẽ nền: bầu trời gradient + mặt đất xanh
    │
    ├── Vẽ nhân vật stick figure tại (x, y):
    │     ├── Idle:       đứng thẳng, tay buông
    │     ├── Raise Hand: tay giơ lên cao
    │     ├── Jump:       tay giơ cao + nhảy lên
    │     ├── Wave:       tay phải vẫy (sin animation)
    │     ├── Running:    chạy nhanh (chân luân phiên)
    │     ├── Walking:    bước chậm (chân luân phiên)
    │     └── Bend:       thân cúi xuống
    │
    └── Vẽ menu overlay (nếu Wave) với hiệu ứng fade-in/out
```

---

### 4.5 Phần 5: UI Overlay (`modules/ui_overlay.py`)

**Class `UIOverlay`** - Hiển thị thông tin hệ thống

```
UIOverlay.draw(frame, action, fps, tracking) → frame
    │
    ├── Vẽ thanh thông tin (nền đen bán trong suốt, 90px trên cùng):
    │
    │   ┌────────────────────────────────────────────┐
    │   │ VR ACTION RECOGNITION              FPS: 28 │
    │   │ Action: ~ Wave                             │
    │   │ Status: Tracking...                        │
    │   └────────────────────────────────────────────┘
    │
    ├── Mỗi action có màu riêng:
    │     Raise Hand: vàng cyan     (0, 255, 255)
    │     Wave:       tím hồng      (255, 100, 255)
    │     Running:    xanh dương    (100, 200, 255)
    │     Walking:    xanh nhạt     (180, 220, 180)
    │     Jump:       xanh non      (150, 255, 150)
    │     Bend:       xanh lá nhạt  (100, 255, 100)
    │     Idle:       xám           (180, 180, 180)
    │     No Person:  đỏ            (0, 0, 255)
    │
    ├── FPS: smoothed (EMA α=0.9) để không nhảy số
    │
    └── Góc phải dưới: "Press ESC to quit"
```

**Debug Overlay** (nhấn D bật/tắt):

```
    ┌──────────────────────────────────────┐
    │  [DEBUG] raw=Wave                    │
    │    L_hand_up: +0.180                 │
    │    R_hand_up: -0.050                 │
    │    bend_h: 1.58 (<1.35=bend)         │
    │    ankle_rng: L:0.012 R:0.008        │
    │    ankle_spd: L:0.0031 R:0.0022      │
    │    wave_rng: 0.045                   │
    │    wave_hold: 12                     │
    └──────────────────────────────────────┘
```

---

## 5. Data Flow (Dòng dữ liệu)

```
Webcam (hardware)
    │
    │  BGR frame (1280x720x3, numpy uint8)
    ▼
┌─────────┐     RGB frame        ┌──────────────────┐
│ Camera  │ ──────────────────── │  PoseEstimator   │
│  .read()│     (cv2.flip)       │  .process()      │
└─────────┘                      │                  │
                                 │  MediaPipe AI:   │
                                 │  detect_for_video│
                                 └────────┬─────────┘
                                          │
                        ┌─────────────────┴──────────────────┐
                        │                                    │
                        ▼                                    ▼
              keypoints dict                        annotated frame
              {                                     (BGR + khung xương)
                "nose": (0.52, 0.25, -0.1, 0.99),
                "left_wrist": (0.7, 0.15, 0.0, 0.95),
                "right_shoulder": (0.4, 0.38, -0.05, 0.98),
                ...14 điểm...
              }
                        │
                        ▼
              ┌──────────────────┐
              │ ActionRecognizer │
              │  .recognize()    │
              │                  │
              │  Kiểm tra theo ưu tiên:  │
              │  Wave → Raise Hand      │
              │  → Running → Walking   │
              │  → Jump → Bend → Idle  │
              └────────┬─────────┘
                       │
                       │  action: str ("Wave")
                       │
             ┌─────────┴──────────┐
             │                    │
             ▼                    ▼
    ┌────────────────┐   ┌──────────────┐
    │ GameController │   │  UIOverlay   │
    │  .update()     │   │  .draw()     │
    │  .draw()       │   │              │
    │                │   │  Action text │
    │  Nhân vật phản │   │  FPS counter │
    │  hồi hành động │   │  Status bar  │
    └───────┬────────┘   └──────┬───────┘
            │                   │
            ▼                   ▼
    game_panel (400x720)   display_frame (1280x720)
            │                   │
            └─────────┬─────────┘
                      │
                      ▼  np.hstack()
              combined frame (1680x720)
                      │
                      ▼
               cv2.imshow()
              ┌────────────────────────────────────────────┐
              │ VR ACTION RECOGNITION  FPS: 28             │
              │ Action: ~ Wave                             │
              │ Status: Tracking...                        │
              │                         ┌────────────────┐ │
              │    [khung xương          │  GAME DEMO     │ │
              │     trên người           │  Action: Wave  │ │
              │     dùng]                │                │ │
              │                         │  ☺ ← nhân vật  │ │
              │                         │  vẫy tay       │ │
              │                         │                │ │
              │                         │  ■■■■ đất ■■■■ │ │
              │ Press ESC to quit       └────────────────┘ │
              └────────────────────────────────────────────┘
              ◄──── Camera 1280px ─────►◄── Game 400px ──►
```

---

## 6. Cấu trúc Project

```
CNĐPT3_N8/
├── main.py                        # Entry point - vòng lặp chính
├── config.py                      # Cấu hình (ngưỡng, màu, camera)
├── requirements.txt               # opencv-python, mediapipe, numpy
├── pose_landmarker_lite.task      # Model AI (tự download lần đầu)
├── doc.md                         # Tài liệu đề tài gốc
├── README.md                      # Tài liệu flow chi tiết (file này)
└── modules/
    ├── __init__.py                # Export 5 class
    ├── camera.py                  # Phần 1: Webcam + hiển thị video
    ├── pose_estimator.py          # Phần 2: MediaPipe nhận diện khung xương
    ├── action_recognizer.py       # Phần 3: Nhận diện hành động
    ├── game_controller.py         # Phần 4: Điều khiển nhân vật game demo
    └── ui_overlay.py              # Phần 5: Giao diện hiển thị kết quả
```

---

## 7. Cài đặt & Chạy

```bash
# Cài thư viện
pip install -r requirements.txt

# Chạy
python main.py
```

- Đứng trước webcam, thực hiện các cử chỉ
- Nhấn **ESC** để thoát
- Nhấn **D** để bật/tắt debug overlay

---

## 8. Bảng hành động hỗ trợ

| Hành động | Cử chỉ người dùng | Thuật toán AI | Phản hồi Game |
|---|---|---|---|
| **Raise Hand** | Giơ 1 hoặc 2 tay lên cao | `wrist.y < shoulder.y - threshold` | Nhân vật giơ tay |
| **Wave** | Giơ tay và vẫy trái-phải | Raise Hand + wrist-x đổi chiều >= 2 + sticky | Mở menu |
| **Running** | Chạy tại chỗ nhanh | Biên độ mắt cá lớn + tốc độ cao + luân phiên 2 chân | Di chuyển nhanh |
| **Walking** | Đi bộ tại chỗ | Luân phiên 2 chân với biên độ/tốc độ vừa | Di chuyển chậm |
| **Jump** | Bật nhảy | Hông nâng lên vượt baseline history | Nhảy cao |
| **Bend** | Cúi người | Tỷ lệ chiều cao thân giảm dưới ngưỡng | Tư thế cúi |

---

## 9. Mở rộng

- Nhận diện hành vi bất thường (an ninh, giám sát)
- Kết hợp AI dự đoán hành động tiếp theo
- Thay thế rule-based bằng LSTM / 3D CNN cho độ chính xác cao hơn
- Kết nối với Unity/Unreal Engine cho VR thực tế
