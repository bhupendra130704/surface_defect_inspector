# Metal Surface Defect Inspector using YOLOV8
Developed a real-time defect inspection system that assists manufacturers in improving product quality by automatically detecting and analyzing surface defects. The model identifies 6 defect types, evaluates their severity, and maintains a structured log of inspections for data-driven decision-making.

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 74.7% |
| Precision | 78.5% |
| Recall | 71.2% |
| Avg Confidence | 0.73 |
| Dataset | NEU-DET (1800 images, 6 classes) |
| Model | YOLOv8m |

## Defect Classes

| Class | Severity | Decision |
|-------|----------|----------|
| inclusion | Critical | AUTO REJECT |
| pitted_surface | Critical | AUTO REJECT |
| scratches | Major | FLAG |
| crazing | Major | FLAG |
| patches | Major | FLAG |
| rolled_in_scale | Major | FLAG |

## System Architecture
```
Camera / Video
      ↓
YOLOv8m Inference
      ↓
ROI Zone Filter  ← only inspect inside defined polygon
      ↓
Severity Classification
      ↓
   REJECT ──→ Reject Signal + Log
   FLAG   ──→ Human Re-check + Log
      ↓
SQLite Database ──→ Streamlit Dashboard
```
---

## How to Run

**1. Clone and install:**
```bash
pip install -r src/requirements.txt
```

**2. Download model weights (49MB):**
[Download best.pt from Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)

Place in `weights/best.pt`
**3. Run on video file:**
```bash
python src/detect.py
```

**4. Run on webcam:**
Change last line in `detect.py`:
```python
run_inspection(source=0)
```

**5. Run on RTSP stream (IP Webcam app):**
```python
run_inspection(source='http://192.168.1.x:8080/video')
```

---
## Training Results

![Results](results/results.png)
![Confusion Matrix](results/confusion_matrix.png)
![PR Curve](results/PR_curve.png)

---

## Dataset

[NEU Surface Defect Dataset](https://universe.roboflow.com) — 
1800 images, 6 defect classes, CC BY 4.0 license.

---
## Built With

`YOLOv8` `OpenCV` `PyTorch` `SQLite` `Streamlit` `FastAPI` `Docker`
