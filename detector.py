import os
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO
from scientific_discriminator import scientific_engine

# Pre-defined vibrant neon palette for bounding boxes and labels
COLOR_PALETTE = [
    (0, 240, 255),    # Electric Cyan
    (255, 60, 180),   # Neon Magenta
    (50, 255, 120),   # Lime Emerald
    (255, 190, 0),    # Vibrant Amber
    (160, 80, 255),   # Violet Purple
    (255, 80, 80),    # Coral Red
    (0, 200, 255),    # Sky Blue
    (255, 140, 0),    # Tangerine
    (130, 255, 240),  # Mint
    (255, 230, 80),   # Lemon
]

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Returns a BGR color tuple for a given class ID."""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

def hex_color(bgr: Tuple[int, int, int]) -> str:
    """Converts BGR tuple to HEX string #RRGGBB."""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

class ObjectDetector:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        self.loaded_models: Dict[str, YOLO] = {}
        self.active_model_name: str = ""
        self.current_model: Optional[YOLO] = None

        # Prioritize Real-Life Room Detector, else Oxford Visual Universe, else 10k universe
        if os.path.exists(os.path.join(self.models_dir, "real_life_room_detector.pt")):
            default_model = "real_life_room_detector.pt"
        elif os.path.exists(os.path.join(self.models_dir, "oxford_universe_detector.pt")):
            default_model = "oxford_universe_detector.pt"
        elif os.path.exists(os.path.join(self.models_dir, "universe_10000_mega_detector.pt")):
            default_model = "universe_10000_mega_detector.pt"
        elif os.path.exists(os.path.join(self.models_dir, "precision_expert_detector.pt")):
            default_model = "precision_expert_detector.pt"
        else:
            default_model = "general_yolo11n.pt"
            
        self.load_model(default_model)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Scans the models directory for .pt files and returns their metadata."""
        models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith(".pt"):
                    file_path = os.path.join(self.models_dir, file)
                    is_general = (file == "general_yolo11n.pt")
                    if file == "real_life_room_detector.pt":
                        display_name = "🏠 Real-Life Room & Everyday Detector (User Room & Background)"
                    elif file == "oxford_universe_detector.pt":
                        display_name = "📘 Oxford Dictionary Visual Model (14,000+ Items)"
                    elif file == "universe_10000_mega_detector.pt":
                        display_name = "🌌 Universe 10,000 Mega Detector (10,000 Objects)"
                    elif is_general:
                        display_name = "General Detector (COCO 80 classes)"
                    else:
                        display_name = file.replace(".pt", "").replace("_", " ").title()

                    models.append({
                        "id": file,
                        "name": display_name,
                        "is_general": is_general,
                        "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                        "is_active": file == self.active_model_name
                    })
        return sorted(models, key=lambda m: (not ("real_life" in m["id"] or "oxford" in m["id"] or "10000" in m["id"]), not m["is_general"], m["name"]))

    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Loads or switches to a model by filename, checking mtime for automatic reload."""
        model_path = os.path.join(self.models_dir, model_name) if not os.path.isabs(model_name) else model_name
        if not os.path.exists(model_path):
            if os.path.exists(model_name):
                model_path = model_name
            else:
                return False

        mtime = os.path.getmtime(model_path)
        cached_info = self.loaded_models.get(model_name)
        if not force_reload and isinstance(cached_info, dict) and cached_info.get("mtime") == mtime:
            self.current_model = cached_info["model"]
            self.active_model_name = model_name
            return True

        try:
            model = YOLO(model_path)
            self.loaded_models[model_name] = {"model": model, "mtime": mtime}
            self.current_model = model
            self.active_model_name = os.path.basename(model_path)
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False

    def get_classes(self) -> Dict[int, str]:
        """Returns the dictionary of class IDs to class names for the active model."""
        if self.current_model and hasattr(self.current_model, "names"):
            return self.current_model.names
        return {}

    def get_taxonomy(self) -> Dict[str, Any]:
        """Loads and returns taxonomy sections if available."""
        tax_path = os.path.join(self.models_dir, "universe_10000_taxonomy.json")
        if os.path.exists(tax_path):
            try:
                import json
                with open(tax_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"sections": {"all": {"name": "All Objects", "count": len(self.get_classes()), "classes": []}}}

    def get_section_classes(self, section: str) -> Optional[set]:
        """Returns the set of class names belonging to a taxonomy section."""
        if not section or section == "all":
            return None
        taxonomy = self.get_taxonomy()
        sec_data = taxonomy.get("sections", {}).get(section)
        if sec_data and sec_data.get("classes"):
            return set(c.lower() for c in sec_data["classes"])
        return None

    def get_oxford_knowledge(self) -> Dict[str, Any]:
        """Loads and caches the Oxford Visual Knowledge Base."""
        if getattr(self, "_oxford_kb", None) is not None:
            return self._oxford_kb
        kb_path = os.path.join(self.models_dir, "oxford_visual_knowledge.json")
        if os.path.exists(kb_path):
            try:
                import json
                with open(kb_path, "r", encoding="utf-8") as f:
                    self._oxford_kb = json.load(f)
                    return self._oxford_kb
            except Exception as e:
                print(f"Error loading Oxford visual KB: {e}")
        self._oxford_kb = {"total_items": 0, "items": {}}
        return self._oxford_kb

    def get_oxford_item_info(self, item_name: str) -> Dict[str, str]:
        """Returns the Oxford Dictionary definition, visual looks, and category for an item."""
        kb = self.get_oxford_knowledge()
        items = kb.get("items", {})
        key = item_name.lower().strip()
        if key in items:
            return items[key]
        for k, v in items.items():
            if k == key or key in k or k in key:
                return v
        return {
            "name": item_name,
            "definition": f"A physical artifact or object recognized as {item_name}.",
            "looks": f"Standard physical visual appearance and features characteristic of a {item_name}.",
            "category": "general"
        }

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        filter_classes: Optional[List[int]] = None,
        section: Optional[str] = None,
        scientific_mode: bool = True
    ) -> Dict[str, Any]:
        """Runs object detection on a BGR image with 50% minimum confidence filtering and Oxford visual knowledge enrichment."""
        if self.current_model is None:
            raise RuntimeError("No model is loaded")

        start_time = time.perf_counter()
        
        section_filter = self.get_section_classes(section)
        effective_classes = filter_classes
        if section_filter is not None and hasattr(self.current_model, "names") and self.current_model.names:
            name_to_id = {str(name).lower(): int(cid) for cid, name in self.current_model.names.items()}
            sec_ids = [name_to_id[c] for c in section_filter if c in name_to_id]
            if sec_ids:
                if effective_classes is not None:
                    effective_classes = [cid for cid in effective_classes if cid in sec_ids]
                else:
                    effective_classes = sec_ids

        # Run inference
        results = self.current_model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=effective_classes,
            verbose=False
        )

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 1)

        detections = []
        summary: Dict[str, int] = {}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.current_model.names.get(cls_id, f"Class {cls_id}")

                    # If section filter is active, skip classes outside this section
                    if section_filter is not None and cls_name.lower() not in section_filter:
                        continue

                    # Scientific Dual-Stage Disambiguation:
                    # Resolves marker vs pen vs pencil, phone vs tv, headset vs headphone, perfume vs water bottle
                    is_verified = False
                    rationale = ""
                    if scientific_mode:
                        verified_name, verified_conf, rationale = scientific_engine.disambiguate(
                            image_bgr=image,
                            box=xyxy,
                            candidate_class=cls_name,
                            confidence=conf
                        )
                        if verified_name.lower() != cls_name.lower():
                            cls_name = verified_name
                            conf = verified_conf
                            is_verified = True
                        elif "Scientific" in rationale:
                            conf = max(conf, verified_conf)
                            is_verified = True

                    # Minimum confidence enforcement (50% min default for accurate answers)
                    if conf < conf_threshold:
                        continue

                    # Retrieve Oxford Dictionary Visual Knowledge ("Their Looks")
                    ox_info = self.get_oxford_item_info(cls_name)

                    color_bgr = get_class_color(cls_id)
                    color_hex = hex_color(color_bgr)

                    detections.append({
                        "box": [round(coord, 1) for coord in xyxy],
                        "confidence": round(conf, 3),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "color": color_hex,
                        "scientific_verified": is_verified,
                        "scientific_rationale": rationale,
                        "oxford_definition": ox_info.get("definition", ""),
                        "oxford_looks": ox_info.get("looks", ""),
                    })

                    summary[cls_name] = summary.get(cls_name, 0) + 1

        return {
            "model": self.active_model_name,
            "inference_time_ms": inference_time_ms,
            "fps": round(1000 / max(inference_time_ms, 1), 1),
            "total_detections": len(detections),
            "summary": summary,
            "detections": detections,
            "image_height": image.shape[0],
            "image_width": image.shape[1],
        }

    def render_boxes(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws polished bounding boxes and badges on the image."""
        output = image.copy()
        h, w = output.shape[:2]

        # Calculate dynamic font scale and line thickness based on image size
        base_dim = max(h, w)
        thickness = max(2, int(base_dim / 350))
        font_scale = max(0.45, base_dim / 1100)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            color = get_class_color(cls_id)

            # 1. Subtle semi-transparent bounding box fill
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)

            # 2. Main crisp bounding box border
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # 3. Corner accents for modern HUD look
            corner_len = min(20, max(8, int((x2 - x1) * 0.15)), int((y2 - y1) * 0.15))
            accent_thickness = thickness + 2
            # Top-left
            cv2.line(output, (x1, y1), (x1 + corner_len, y1), color, accent_thickness)
            cv2.line(output, (x1, y1), (x1, y1 + corner_len), color, accent_thickness)
            # Top-right
            cv2.line(output, (x2, y1), (x2 - corner_len, y1), color, accent_thickness)
            cv2.line(output, (x2, y1), (x2, y1 + corner_len), color, accent_thickness)
            # Bottom-left
            cv2.line(output, (x1, y2), (x1 + corner_len, y2), color, accent_thickness)
            cv2.line(output, (x1, y2), (x1, y2 - corner_len), color, accent_thickness)
            # Bottom-right
            cv2.line(output, (x2, y2), (x2 - corner_len, y2), color, accent_thickness)
            cv2.line(output, (x2, y2), (x2, y2 - corner_len), color, accent_thickness)

            # 4. Badge label with background
            badge_prefix = "🔬 " if det.get("scientific_verified") else ""
            label = f"{badge_prefix}{cls_name} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Position badge above box, or inside if too close to top
            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
            badge_x2 = min(w, x1 + text_w + 14)

            # Badge background
            cv2.rectangle(output, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Badge text (dark text for bright neon badges)
            text_pos_y = badge_y2 - 5 if y1 - text_h - 10 >= 0 else badge_y2 - baseline - 2
            cv2.putText(output, label, (x1 + 6, text_pos_y), font, font_scale, (15, 15, 20), 1, cv2.LINE_AA)

        return output
