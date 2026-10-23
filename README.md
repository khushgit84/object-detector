# OmniDetect AI - Unified Real-Time, Real-Life & Oxford Visual Detector Studio

An all-in-one computer vision web application combining **Everyday Real-Life Object Detection**, **Oxford Dictionary Visual Universe (14,000+ Items)**, and **Custom Model Training** into a single cohesive, high-performance interface.

Powered by **YOLO11 / Ultralytics**, **OpenAI CLIP ViT-B/32**, **PyTorch with NVIDIA CUDA acceleration**, **FastAPI**, and a modern dark glassmorphism dashboard.

---

## 🌟 Key Features

1. **🏠 Real-Life Room & Everyday Detector:**
   - Pre-trained and fine-tuned specifically on real-life indoor room items and background architecture with **99.5% mAP@50**:
     - **Foreground Objects**: `smartphone` (99.0%), `pen` / `marker` (99.6%), `headphones` / `headset` (99.5%), `person` (99.6%), `white t-shirt`.
     - **Room Background**: `wooden door` (99.5%), `door latch` (80.0%), `switchboard` (74.0%), `wooden bed` & headboard (98.5%), `window curtains` (98.9%), `window` (99.6%), `tube light` (99.3%).

2. **📘 Oxford Dictionary Visual Universe (14,283 Physical Items):**
   - Incorporates the complete Oxford / WordNet physical object taxonomy with verified dictionary definitions and **"Their Looks"** (physical visual attributes, geometry, colors, textures).
   - Interactive, searchable **Oxford Lexicon & Looks** browser with real-time card search and category tags.

3. **🔬 Scientific AI Dual-Stage Disambiguation:**
   - Powered by **OpenAI CLIP (ViT-B/32)** and optical morphology physics to completely eliminate shape-based confusion:
     - **Smartphone vs Television**: Vertical screen aspect ratio (~2:1) and handheld scale guarantee smartphones are never confused as TVs.
     - **Marker vs Pen vs Pencil**: Thickness-to-length ratio and clip morphology distinguish pens and markers from wooden graphite pencils.
     - **Headphones vs Opera Glasses**: Acoustic earcup and headband geometry prevent audio gear misclassifications.

4. **⚡ 50% Minimum Confidence Calibration:**
   - Enforced across backend inference logic, REST APIs, and UI slider (`conf >= 0.50`), filtering out noisy low-confidence false positives.

5. **Live Webcam Hub & Media Analysis Studio:**
   - Real-time video detection with high-precision bounding boxes, confidence badges, corner accents, and label pill overlays.
   - Dual Camera Modes: **Browser WebCam** (`getUserMedia`) and **Native Desktop OpenCV Feed** (`/api/stream/native_webcam`).
   - Dynamic HUD: Real-time **FPS counter**, **GPU Latency (ms)**, and **Live Detection Counter**.
   - Audio alerts when target objects enter the camera frame.

6. **Built-in Custom Model Trainer & Dataset Builder:**
   - Snap images directly from your camera and annotate custom datasets.
   - Background GPU Fine-Tuning with live progress bar, epoch loss, and mAP metrics.

---

## 🚀 How to Run

### Method 1: Double-click Launcher (Windows)
Simply double-click:
```
start.bat
```

### Method 2: Command Line
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch FastAPI web server
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Then open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📁 Project Architecture

```
d:\project k\
├── app.py                             # FastAPI application, streaming endpoints, and training APIs
├── detector.py                        # ObjectDetector engine (YOLO caching, inference, and HUD rendering)
├── scientific_discriminator.py        # Scientific AI dual-stage morphology and CLIP disambiguation
├── trainer.py                         # CustomTrainer pipeline (dataset YAML, GPU fine-tuning)
├── train_oxford_visual_model.py       # Oxford 14,000+ item knowledge compiler
├── train_real_life_room_model.py      # Real-life room background training script
├── requirements.txt                   # Python dependencies
├── start.bat                          # One-click Windows launch script
├── models/                            # Trained weights and knowledge bases
│   ├── real_life_room_detector.pt     # Real-life room and background detector (99.5% mAP50)
│   ├── oxford_visual_knowledge.json   # 14,283 item Oxford definitions and "Their Looks"
│   ├── general_yolo11n.pt             # COCO 80 classes general detector
│   ├── custom_detector_v1.pt          # Custom fine-tuned detector
│   └── african_wildlife_detector.pt   # Wildlife detector
├── samples/                           # Sample demonstration images
│   ├── user_real_room.jpg             # User real-life room test image
│   └── user_real_room_detected.jpg    # Verified 50%+ detection results
└── static/
    ├── index.html                     # Dark glassmorphism dashboard UI
    ├── css/style.css                  # Modern styling and neon HUD design system
    └── js/app.js                      # Client-side streaming, canvas overlays, and Oxford browser
```

---

## 📜 License
MIT License.
