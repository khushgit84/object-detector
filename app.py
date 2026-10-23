import os
import io
import base64
import time
import json
import cv2
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detector import ObjectDetector
from trainer import CustomTrainer

app = FastAPI(title="OmniDetect AI - Unified General & Custom Object Detector")

# Allow CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector & Trainer
detector = ObjectDetector(models_dir="models")
trainer = CustomTrainer(datasets_dir="datasets", models_dir="models")

# Mount static, models, and samples folders
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("samples", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# ----------------- Model Management -----------------

@app.get("/api/models")
async def list_models():
    """Returns available general and custom models."""
    return {
        "models": detector.get_available_models(),
        "active_model": detector.active_model_name
    }

@app.post("/api/models/select")
async def select_model(payload: dict):
    """Switches the active model."""
    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    success = detector.load_model(model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Failed to load model {model_name}")

    return {
        "status": "success",
        "active_model": detector.active_model_name,
        "classes": detector.get_classes()
    }

@app.post("/api/models/upload")
async def upload_custom_model(file: UploadFile = File(...)):
    """Uploads a custom .pt YOLO model file directly."""
    if not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt weights files are supported")

    dest_path = os.path.join(detector.models_dir, file.filename)
    with open(dest_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    detector.load_model(file.filename)
    return {
        "status": "success",
        "message": f"Model {file.filename} uploaded and activated successfully",
        "active_model": detector.active_model_name,
        "classes": detector.get_classes()
    }

@app.get("/api/classes")
async def get_classes():
    """Returns the classes of the current active model."""
    return {"classes": detector.get_classes()}

@app.get("/api/taxonomy")
async def get_taxonomy():
    """Returns taxonomy sections and category mappings for the 10,000+ universe."""
    return detector.get_taxonomy()

@app.get("/api/oxford/item/{item_name}")
async def get_oxford_item(item_name: str):
    """Returns the Oxford Dictionary definition and visual looks for an item."""
    return detector.get_oxford_item_info(item_name)

@app.get("/api/oxford/knowledge")
async def get_oxford_knowledge():
    """Returns summary of the Oxford Visual Knowledge Base."""
    kb = detector.get_oxford_knowledge()
    return {
        "total_items": kb.get("total_items", 0),
        "min_confidence_filter": kb.get("min_confidence_filter", 0.50),
        "compiled_at": kb.get("compiled_at", "")
    }

# ----------------- Inference Endpoints -----------------

@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.50),
    iou: float = Form(0.45),
    classes: Optional[str] = Form(None),
    section: Optional[str] = Form(None),
    scientific_mode: bool = Form(True)
):
    """Processes an uploaded image and returns detection results with annotated image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        filter_classes = [int(c) for c in classes.split(",")] if classes and classes.strip() else None

        results = detector.predict(
            img,
            conf_threshold=conf,
            iou_threshold=iou,
            filter_classes=filter_classes,
            section=section,
            scientific_mode=scientific_mode
        )
        annotated_img = detector.render_boxes(img, results["detections"])

        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        results["annotated_image"] = f"data:image/jpeg;base64,{annotated_b64}"
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/frame")
async def detect_frame(payload: dict):
    """Fast detection endpoint for base64 camera frames with section filter support."""
    try:
        data_url = payload.get("image")
        conf = float(payload.get("conf", 0.50))
        iou = float(payload.get("iou", 0.45))
        classes_str = payload.get("classes")
        section = payload.get("section")
        scientific_mode = payload.get("scientific_mode", True)
        return_image = payload.get("return_image", True)

        if not data_url:
            raise HTTPException(status_code=400, detail="Image data is missing")

        # Strip header if present
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        image_bytes = base64.b64decode(data_url)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode image frame")

        filter_classes = [int(c) for c in classes_str.split(",")] if classes_str and str(classes_str).strip() else None

        results = detector.predict(
            img,
            conf_threshold=conf,
            iou_threshold=iou,
            filter_classes=filter_classes,
            section=section,
            scientific_mode=scientific_mode
        )

        if return_image:
            annotated_img = detector.render_boxes(img, results["detections"])
            _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            results["annotated_image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- WebSocket for Real-Time Streaming -----------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """High-throughput WebSocket for real-time video streaming & detection."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)

            frame_b64 = req.get("frame")
            if not frame_b64:
                continue

            conf = float(req.get("conf", 0.25))
            iou = float(req.get("iou", 0.45))
            classes = req.get("classes")

            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                results = detector.predict(img, conf_threshold=conf, iou_threshold=iou, filter_classes=classes)
                # Send back pure JSON box data for zero-overhead client-side rendering
                await websocket.send_text(json.dumps(results))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

# ----------------- Native Local Camera MJPEG Stream -----------------

def gen_local_webcam_stream(conf=0.25):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = detector.predict(frame, conf_threshold=conf)
        annotated = detector.render_boxes(frame, results["detections"])

        ret, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

@app.get("/api/stream/native_webcam")
def native_webcam_feed(conf: float = 0.25):
    """Direct OpenCV webcam feed for desktop local cameras."""
    return StreamingResponse(
        gen_local_webcam_stream(conf=conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ----------------- Custom Model Training & Dataset Studio -----------------

@app.get("/api/datasets")
async def list_datasets():
    """Returns available custom datasets."""
    return {"datasets": trainer.list_datasets()}

@app.post("/api/dataset/create")
async def create_dataset(payload: dict):
    """Creates a new custom dataset structure."""
    name = payload.get("name", "").strip()
    classes = payload.get("classes", [])

    if not name or not classes:
        raise HTTPException(status_code=400, detail="Dataset name and at least one class are required")

    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).lower()
    yaml_path = trainer.prepare_dataset(safe_name, classes)

    return {
        "status": "success",
        "dataset_name": safe_name,
        "classes": classes,
        "yaml_path": yaml_path
    }

@app.post("/api/dataset/add_sample")
async def add_dataset_sample(
    dataset_name: str = Form(...),
    class_id: int = Form(...),
    box_normalized: str = Form(...),  # "cx,cy,w,h" in 0.0-1.0 coords
    image: UploadFile = File(...)
):
    """Saves an image and YOLO format label to the dataset."""
    dataset_path = os.path.join(trainer.datasets_dir, dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset does not exist")

    # 80/20 train/val split
    timestamp = int(time.time() * 1000)
    split = "train" if (timestamp % 5 != 0) else "val"

    img_filename = f"sample_{timestamp}.jpg"
    lbl_filename = f"sample_{timestamp}.txt"

    img_dest = os.path.join(dataset_path, "images", split, img_filename)
    lbl_dest = os.path.join(dataset_path, "labels", split, lbl_filename)

    # Save image
    contents = await image.read()
    with open(img_dest, "wb") as f:
        f.write(contents)

    # Save YOLO format label: <class_id> <x_center> <y_center> <width> <height>
    cx, cy, w, h = [float(v) for v in box_normalized.split(",")]
    with open(lbl_dest, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return {
        "status": "success",
        "split": split,
        "file": img_filename
    }

@app.post("/api/dataset/create_sample_demo")
async def create_sample_demo_dataset():
    """Generates an instant sample dataset (e.g. geometric badges/symbols) for testing the custom training pipeline."""
    dataset_name = "demo_shapes_detector"
    class_names = ["gold_star", "cyan_badge", "neon_triangle"]
    trainer.prepare_dataset(dataset_name, class_names)
    dataset_path = os.path.join(trainer.datasets_dir, dataset_name)

    # Generate synthetic training samples
    for i in range(24):
        split = "train" if i < 20 else "val"
        canvas = np.random.randint(20, 60, (416, 416, 3), dtype=np.uint8)

        # Pick a class
        cls_id = i % 3
        # Pick random location
        x = np.random.randint(60, 300)
        y = np.random.randint(60, 300)
        size = np.random.randint(40, 70)

        x1, y1 = max(10, x - size), max(10, y - size)
        x2, y2 = min(406, x + size), min(406, y + size)

        # Draw shape based on class
        if cls_id == 0:  # Gold Star / Circle
            cv2.circle(canvas, (x, y), size, (0, 215, 255), -1)
            cv2.circle(canvas, (x, y), size // 2, (255, 255, 255), -1)
        elif cls_id == 1:  # Cyan Badge / Rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 240, 0), -1)
            cv2.rectangle(canvas, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), (180, 100, 0), -1)
        else:  # Neon Triangle
            pts = np.array([[x, y1], [x1, y2], [x2, y2]], np.int32)
            cv2.fillPoly(canvas, [pts], (0, 255, 120))

        # Save synthetic image
        img_name = f"synth_{cls_id}_{i}.jpg"
        lbl_name = f"synth_{cls_id}_{i}.txt"
        cv2.imwrite(os.path.join(dataset_path, "images", split, img_name), canvas)

        # YOLO coords: cx, cy, w, h normalized
        cx = (x1 + x2) / (2.0 * 416)
        cy = (y1 + y2) / (2.0 * 416)
        w = (x2 - x1) / 416.0
        h = (y2 - y1) / 416.0

        with open(os.path.join(dataset_path, "labels", split, lbl_name), "w") as f:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return {
        "status": "success",
        "message": "Demo dataset 'demo_shapes_detector' generated with 24 annotated images!",
        "dataset_name": dataset_name,
        "classes": class_names
    }

@app.post("/api/train/start")
async def start_training(payload: dict):
    """Triggers custom model fine-tuning."""
    dataset_name = payload.get("dataset_name")
    output_model_name = payload.get("output_model_name", "custom_detector")
    epochs = int(payload.get("epochs", 5))

    if not dataset_name:
        raise HTTPException(status_code=400, detail="dataset_name is required")

    success = trainer.start_training(
        dataset_name=dataset_name,
        output_model_name=output_model_name,
        epochs=epochs,
        imgsz=416,
        batch_size=8
    )

    if not success:
        raise HTTPException(status_code=409, detail="A training job is already currently in progress")

    return {"status": "started", "job": trainer.get_status()}

@app.get("/api/train/status")
async def training_status():
    """Polls training status and metrics."""
    return trainer.get_status()

if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print("🚀 OmniDetect AI - Real-time & Custom Object Detector")
    print("📍 Running at: http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
