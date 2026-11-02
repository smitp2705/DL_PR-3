# CV_PR3 — Classical Image Processing → Face Detection → YOLO Object Detection

**Deep Learning · Project PR 3**

An integrated, real-time computer-vision pipeline built with **OpenCV 5.0**, **YuNet** and **Ultralytics YOLOv8**. The project moves from classical, hand-engineered image processing (morphology, bitwise logic, histograms) to modern deep-learning detectors (face + objects), and finally combines everything into one webcam pipeline. This mirrors how low-cost smart-surveillance and retail-analytics systems are built: cheap classical pre-processing cleans the frame, and deep detectors provide accurate identification on top.

---

## Table of Contents

1. [Video Explanation](#-video-explanation)
2. [Project Overview](#-project-overview)
3. [Data & Model Sources](#-data--model-sources)
4. [Setup & Preprocessing Steps](#-setup--preprocessing-steps)
5. [Repository Structure](#-repository-structure)
6. [Topics Covered](#-topics-covered)
7. [Results & Plots](#-results--plots)
8. [Final Comparison Table](#-final-comparison-table)
9. [Key Learnings & Reflection](#-key-learnings--reflection)
10. [Tools & Technologies](#-tools--technologies)
11. [How to Run](#-how-to-run)
12. [License & Acknowledgements](#-license--acknowledgements)

---

## 🎥 Video Explanation

📺 **Watch here:** [CV_PR3_YourName_GRID.mp4](PASTE_YOUR_GOOGLE_DRIVE_OR_YOUTUBE_LINK_HERE)

A 5–10 minute walkthrough (face picture-in-picture + screen) covering all 5 tasks, the live webcam face-detection and YOLO demos, the FPS chart, and the final results table. It explains verbally:

1. Erosion/dilation as min/max filters, and why opening vs closing is chosen
2. How alpha/beta affect the histogram, brightness and contrast
3. YuNet's CNN-based detection, landmarks, and the score-threshold trade-off
4. YOLO's single-stage grid + NMS design and the accuracy/speed trade-off of model size

---

## 📌 Project Overview

| Field | Detail |
|---|---|
| Institute | Red & White Skill Education |
| Subject | Deep Learning |
| Project | PR 3 |
| Task Type | Classical Image Processing → Face Detection → Deep-Learning Object Detection (integrated real-time CV pipeline) |
| Framework | OpenCV (cv2) 5.0.0 · Ultralytics YOLOv8 · NumPy · Matplotlib (Python 3.x, Jupyter Notebook) |
| Duration | 3 days (2 hours per day) |
| Total Marks | 10 |

---

## 📂 Data & Model Sources

This project uses three complementary sources rather than a single labelled dataset.

| Source | Used For | Link |
|---|---|---|
| OpenCV sample data (`lena.jpg`, `baboon.jpg`, `coins.png`, etc.) | Morphology, bitwise operations, histograms | https://github.com/opencv/opencv/tree/master/samples/data |
| Personal phone photos (2–3 images) | Histograms, face and object detection on varied pose/lighting | _Own images (not redistributed)_ |
| **YuNet** — `face_detection_yunet_2023mar.onnx` | Face detection via `cv2.FaceDetectorYN_create()` | https://github.com/opencv/opencv_zoo |
| **YOLOv8-Nano** — `yolov8n.pt` (COCO, 80 classes) | Object detection (auto-downloads on first use) | https://github.com/ultralytics/ultralytics |
| Live input | Laptop webcam (`cv2.VideoCapture(0)`), or a pre-recorded video if no webcam is available | — |

**Licenses:** OpenCV sample data — BSD-3-Clause · Ultralytics YOLOv8 — AGPL-3.0 (pretrained weights free for academic/educational use).

**Image pack (6–10 images):** one noisy/textured image for morphology, one image with 2–3 solid-colour objects on a plain background for bitwise masking, and 3–4 general photos with people and everyday objects (person, car, bottle, chair) for histograms and detection.

---

## ⚙️ Setup & Preprocessing Steps

1. **Install dependencies** (once):
   ```bash
   pip install opencv-python==5.0.0.* opencv-contrib-python==5.0.0.* ultralytics matplotlib numpy
   ```
2. **Confirm OpenCV version** inside the notebook: `print(cv2.__version__)` → should print `5.0.0`.
3. **Folder layout:** keep static images in `data/images/` and model weights (including `face_detection_yunet_2023mar.onnx`) in `data/models/` so paths stay consistent.
4. **Download YuNet weights** from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) into `data/models/`. `yolov8n.pt` downloads automatically on first use.
5. **Image loading & preprocessing:**
   - Load one grayscale and one colour image with `cv2.imread()`; verify `.shape` and `.dtype`.
   - Convert BGR → RGB for Matplotlib display via a `show(img, title)` helper (since `cv2.imshow` does not render inline in Jupyter).
   - Convert to grayscale and binarize with `cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)` to create input for morphological experiments.
6. **Webcam note:** if no webcam is available, substitute a pre-recorded video in `cv2.VideoCapture(...)`.

---

## 🗂️ Repository Structure

```
CV_PR3/
├── CV_PR3.ipynb              # Main notebook (all 5 tasks, Restart & Run All clean)
├── CV_PR3.html               # HTML export of the notebook
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── data/
│   ├── images/               # Static test images
│   └── models/               # YuNet ONNX + YOLO weights
└── plots/
    ├── morphology_grid.png
    ├── histogram_comparison.png
    ├── face_detection_comparison.png
    ├── yolo_detections.png
    ├── fps_comparison.png
    └── results_table.png
```

---

## 🧠 Topics Covered

### 1️⃣ Task 1 — Environment Setup, Image I/O & Morphological Operations
Applied erosion, dilation, opening and closing on a binarized image using rectangular kernels, then compared kernel shapes (RECT / ELLIPSE / CROSS) and sizes (3×3 / 5×5 / 9×9) on a noisy binary image.
Erosion acts as a local-minimum filter and dilation as a local-maximum filter; opening removes small bright noise while closing fills small holes — useful for cleaning masks before a detector.

### 2️⃣ Task 2 — Bitwise Operations & Image Histograms
Demonstrated AND / OR / XOR / NOT on circle and rectangle masks, then used them for a real masked-ROI extraction and logo/watermark overlay. Plotted grayscale and per-channel B/G/R histograms and adjusted brightness/contrast with `cv2.convertScaleAbs`.
Using `new_pixel = alpha·pixel + beta`, alpha stretches/compresses the histogram (contrast) and beta shifts it (brightness), making histograms a quick diagnostic for exposure problems.

### 3️⃣ Task 3 — Face Detection with YuNet (OpenCV 5.0)
Ran the `cv2.FaceDetectorYN` YuNet detector on static images (varied pose, lighting, profile and occluded faces) and on the live webcam, drawing boxes, 5 facial landmarks and confidence scores. Ran a score-threshold experiment (0.5 / 0.7 / 0.9) and built a face-region Gaussian-blur privacy filter.
YuNet is a compact CNN detector that returns boxes, landmarks and confidence in a single forward pass; the score threshold trades missed detections against false positives.

### 4️⃣ Task 4 — Object Detection with YOLO
Ran pretrained YOLOv8n (COCO, 80 classes) on static images and the webcam, reporting FPS. Tuned `conf` (0.25 / 0.5 / 0.75) and `iou` (0.3 / 0.5 / 0.7), showed a duplicate-box example caused by low IoU, and built a per-class detection summary bar chart.
YOLO is a single-stage, grid-based detector that regresses boxes and class probabilities in one pass, followed by Non-Max Suppression; the nano variant favours speed over accuracy compared with s/m/l/x.

### 5️⃣ Task 5 — Integrated Pipeline & Final Comparison
Combined YuNet (green boxes) and YOLO (red boxes with labels) on a single webcam stream, tested optional morphological pre-cleaning on noisy frames, and benchmarked FPS for face-only, YOLO-only and combined pipelines over 100 frames.
Built a full comparison table across all techniques and closed with a reflection on the best classical + deep-learning trade-off for edge vs cloud deployment.

---

## 📊 Results & Plots

> Add the actual figures to `/plots` and keep the captions below. Replace the _TBD_ values with your measured results.

### Morphological Operations
![Morphology Grid](plots/morphology_grid.png)
*Figure 1 — Original binary image with erosion, dilation, opening and closing (1×5 grid), plus the 3×3 kernel shape × size comparison.*

### Histograms & Brightness/Contrast
![Histogram Comparison](plots/histogram_comparison.png)
*Figure 2 — Original, brightened (α=1.5, β=30) and darkened (α=0.7, β=−30) images with their grayscale histograms stacked below.*

### Face Detection (YuNet)
![Face Detection Comparison](plots/face_detection_comparison.png)
*Figure 3 — YuNet detections with boxes, landmarks and confidence scores at thresholds 0.5 / 0.7 / 0.9, and the face-blur privacy filter.*

### YOLO Object Detection
![YOLO Detections](plots/yolo_detections.png)
*Figure 4 — YOLOv8n detections with class labels and confidence, plus the confidence/IoU threshold experiment and per-class detection summary.*

### FPS Benchmark
![FPS Comparison](plots/fps_comparison.png)
*Figure 5 — Average FPS over a 100-frame sample for face detection alone, YOLO alone, and the combined pipeline.*

| Configuration | Average FPS |
|---|---|
| YuNet face detection only | _TBD_ |
| YOLOv8n only | _TBD_ |
| Combined pipeline | _TBD_ |

### Experiment Observations

| Experiment | Observation |
|---|---|
| Best kernel (shape × size) for salt-and-pepper noise | _TBD_ |
| YuNet threshold 0.5 / 0.7 / 0.9 | _TBD (faces detected, false positives)_ |
| YOLO conf 0.25 / 0.5 / 0.75 | _TBD (boxes retained)_ |
| YOLO IoU 0.3 / 0.5 / 0.7 | _TBD (duplicate boxes)_ |
| With vs without morphological pre-cleaning | _TBD_ |

---

## 📋 Final Comparison Table

![Results Table](plots/results_table.png)

| Technique | Type | What It Detects / Does | Lighting Robustness | Approx. FPS | Best Use Case |
|---|---|---|---|---|---|
| Morphology (erode / dilate / open / close) | Classical | Cleans binary masks: removes noise, fills holes | _TBD_ | _TBD_ | Mask cleanup before a downstream detector |
| Bitwise ops & histograms | Classical | Region masking/compositing; exposure & contrast diagnostics | _TBD_ | _TBD_ | ROI extraction, watermarking, deciding on pre-processing |
| YuNet face detection | Deep Learning | Faces + 5 landmarks + confidence | _TBD_ | _TBD_ | Real-time face detection, privacy blurring |
| YOLOv8n | Deep Learning | 80 COCO object classes | _TBD_ | _TBD_ | Real-time multi-object detection |

---

## 💡 Key Learnings & Reflection

_Summarise your own conclusions here (from the notebook's closing Markdown cell):_

- Which combination of classical pre-processing + deep-learning detection gave the best accuracy-to-compute trade-off for real-time monitoring.
- What you would change for a **low-power edge device** (e.g. keep `yolov8n`, raise the YuNet `score_threshold`, lighter pre-processing).
- What you would change for a **cloud server** (e.g. move to `yolov8s/m`, lower thresholds for higher recall).

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.x | Programming language |
| Jupyter Notebook | Interactive development environment |
| OpenCV 5.0.0 (`opencv-python`, `opencv-contrib-python`) | Image processing, morphology, bitwise ops, histograms, YuNet face detection |
| Ultralytics YOLOv8 | Pretrained object detection |
| NumPy | Array and mask operations |
| Matplotlib | Plots, histograms, grids, FPS charts |
| Git & GitHub | Version control and submission |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place images in data/images/ and the YuNet ONNX file in data/models/

# 5. Launch the notebook
jupyter notebook CV_PR3.ipynb
```

Then choose **Kernel → Restart & Run All**. The webcam cells need a connected camera (or swap in a video file).

**`requirements.txt`**
```
opencv-python==5.0.0.*
opencv-contrib-python==5.0.0.*
ultralytics
numpy
matplotlib
```

---

## 📜 License & Acknowledgements

- OpenCV sample data: BSD-3-Clause
- Ultralytics YOLOv8: AGPL-3.0 (pretrained weights free for academic/educational use)
- YuNet model from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- Project brief by **Red & White Skill Education** — _"Shaping Skills for Scaling Higher...!!!"_
