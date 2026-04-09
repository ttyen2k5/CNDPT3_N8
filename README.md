# Action Recognition - Dodge Game

Hệ thống nhận diện hành động (Jump / Bend) bằng AI, điều khiển game né chướng ngại vật qua webcam.

## Yêu cầu

- Python 3.10+
- Webcam

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

### Bước 1: Train model

Mở và chạy toàn bộ `train.ipynb` trong Jupyter Notebook.

```bash
jupyter notebook train.ipynb
```

Notebook sẽ:
- Đọc video `jump_*.mp4` và `bend_*.mp4` trong thư mục `data/`
- Trích xuất keypoints bằng MediaPipe
- Train model RandomForest
- Xuất ra file `model.pkl`

> Lần đầu chạy sẽ tự tải model MediaPipe (`pose_landmarker_lite.task`).

### Bước 2: Chạy game

Khi đã có `model.pkl`:

```bash
python main.py
```

Đứng trước webcam và thực hiện:
- **Nhảy** → nhân vật nhảy né chướng ngại vật trên mặt đất
- **Cúi người** → nhân vật cúi né chướng ngại vật trên cao
- **Đứng yên** → nhân vật đứng bình thường

### Phím tắt

| Phím | Chức năng |
|---|---|
| ESC | Thoát |
| D | Bật/tắt debug overlay |
| R | Restart game |

## Thêm dữ liệu training

1. Quay video hành động nhảy, đặt tên `jump_N.mp4` (N = số thứ tự)
2. Quay video hành động cúi, đặt tên `bend_N.mp4`
3. Đưa vào thư mục `data/`
4. Chạy lại `train.ipynb` để train lại model

Dữ liệu được tự động chia 80% train / 20% test.

## Cấu trúc thư mục

```
├── main.py                 # Chạy game + camera
├── train.ipynb             # Train model
├── config.py               # Cấu hình
├── model.pkl               # Model (output train.ipynb)
├── requirements.txt        # Dependencies
├── data/                   # Video training
│   ├── jump_*.mp4
│   └── bend_*.mp4
└── modules/
    ├── camera.py           # Webcam
    ├── pose_estimator.py   # MediaPipe Pose
    ├── action_recognizer.py# ML inference
    ├── game_controller.py  # Game logic
    └── ui_overlay.py       # UI hiển thị
```
