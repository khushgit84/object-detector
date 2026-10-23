"""
Scientific Neural Discriminator Engine
Applies computer vision science, morphological aspect physics, and fine-grained CLIP
multimodal feature ensembles to disambiguate commonly confused object silhouettes:
1. Writing Instruments: Marker vs. Pen vs. Pencil vs. Highlighter
2. Electronic Screens: Smartphone vs. Television vs. Computer Monitor vs. Laptop vs. Tablet
3. Audio Gear: Headphone vs. Headset (with boom mic) vs. Earbuds
4. Containers: Perfume Bottle vs. Water Bottle vs. Spray Bottle
5. Sports Gear: Baseball Bat vs. Cylindrical Objects
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class ScientificDiscriminator:
    """Two-Stage Neural & Morphological Verification Engine for Object Disambiguation."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.preprocess = None
        self.family_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        self.is_initialized = False

        # Define ambiguous shape families and fine-grained scientific prompt descriptors
        self.families = {
            "writing": {
                "keywords": ["pen", "pencil", "marker", "highlighter", "eraser", "ballpoint", "sharpener"],
                "prompts": {
                    "marker": [
                        "a thick marker pen with a bold plastic barrel and removable cap",
                        "a whiteboard dry erase marker or permanent marker with wide felt tip",
                        "a colored marker for drawing with a wide cylindrical body"
                    ],
                    "pen": [
                        "a slender ballpoint pen, gel pen, or rollerball pen with a fine metal writing tip",
                        "a clicker pen or capped pen for writing with ink and a pocket clip",
                        "a fountain pen or precision ink pen"
                    ],
                    "pencil": [
                        "a wooden graphite pencil with sharpened conical wood tip and black lead point",
                        "a classic yellow wooden pencil with pink rubber eraser collar on top",
                        "a mechanical pencil with thin graphite lead"
                    ],
                    "highlighter": [
                        "a fluorescent highlighter pen with a wide flat body and chisel felt tip",
                        "a neon yellow, green, or pink highlighter pen"
                    ]
                }
            },
            "screens": {
                "keywords": ["tv", "television", "cell phone", "phone", "smartphone", "laptop", "monitor", "screen", "tablet", "ipad"],
                "prompts": {
                    "smartphone": [
                        "a handheld smartphone or iPhone with a tall narrow touchscreen display held in hand",
                        "a mobile cell phone with camera lens and compact handheld proportions",
                        "a modern touchscreen smartphone display"
                    ],
                    "television": [
                        "a large widescreen flat television set mounted on a wall or sitting on a living room TV stand",
                        "a big flat screen TV display in a room",
                        "a television display monitor with wide horizontal aspect ratio"
                    ],
                    "computer monitor": [
                        "a desktop computer monitor sitting on a desk stand next to keyboard and mouse",
                        "a PC gaming computer display"
                    ],
                    "laptop": [
                        "an opened laptop notebook computer with display screen and attached physical keyboard below"
                    ],
                    "tablet": [
                        "a touchscreen iPad or Android tablet computer held in hands"
                    ]
                }
            },
            "audio": {
                "keywords": ["headphone", "headphones", "headset", "earphone", "earphones", "earbuds", "earpiece"],
                "prompts": {
                    "headset": [
                        "a gaming headset or call center headset with a boom microphone arm extending toward the mouth",
                        "over-ear communications headset with an attached microphone boom on the ear cup"
                    ],
                    "headphone": [
                        "audio headphones with over-ear or on-ear cups and a headband without any boom microphone",
                        "music listening studio headphones"
                    ],
                    "earbuds": [
                        "small wireless in-ear earbuds or AirPods fitting inside the ear canal"
                    ]
                }
            },
            "containers": {
                "keywords": ["bottle", "perfume", "cologne", "water bottle", "spray bottle", "flask", "thermos"],
                "prompts": {
                    "perfume bottle": [
                        "a luxury glass perfume bottle or cologne fragrance spray bottle with metallic nozzle atomizer",
                        "an elegant fragrance flask with ornate cap and perfume mist sprayer"
                    ],
                    "water bottle": [
                        "a reusable drinking water bottle, sports flask, thermos, or plastic mineral water bottle",
                        "a tall beverage bottle with a twist cap, drinking spout, or handle"
                    ],
                    "spray bottle": [
                        "a cleaning trigger spray bottle or plastic pump bottle with finger spray nozzle"
                    ]
                }
            },
            "sports": {
                "keywords": ["bat", "baseball bat", "cricket bat", "racket", "club"],
                "prompts": {
                    "baseball bat": [
                        "a wooden or metal baseball bat with a narrow taped handle and thick smooth cylindrical barrel",
                        "a regulation baseball bat for hitting baseballs"
                    ],
                    "tennis racket": [
                        "a tennis racket with an oval string bed and handle"
                    ]
                }
            }
        }

    def initialize(self):
        """Pre-computes normalized text embeddings on CUDA for instantaneous inference."""
        if self.is_initialized or not CLIP_AVAILABLE:
            return

        try:
            t0 = time.time()
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()

            for family_name, family_info in self.families.items():
                self.family_embeddings[family_name] = {}
                for target_class, prompt_list in family_info["prompts"].items():
                    tokens = clip.tokenize(prompt_list).to(self.device)
                    with torch.no_grad():
                        feats = self.model.encode_text(tokens)
                        feats /= feats.norm(dim=-1, keepdim=True)
                        # Average ensemble prompts and re-normalize
                        mean_feat = feats.mean(dim=0, keepdim=True)
                        mean_feat /= mean_feat.norm(dim=-1, keepdim=True)
                        self.family_embeddings[family_name][target_class] = mean_feat

            self.is_initialized = True
            dur = round(time.time() - t0, 3)
            print(f"[ScientificDiscriminator] Initialized on {self.device} with {len(self.families)} families in {dur}s")
        except Exception as e:
            print(f"[ScientificDiscriminator] Error loading CLIP: {e}")

    def find_family(self, class_name: str) -> Optional[str]:
        """Finds which ambiguous family a candidate class belongs to."""
        name_lower = class_name.lower().strip()
        # Audio check first: prevent "headphones" matching "phone"
        if any(a in name_lower for a in ["headphone", "headphones", "headset", "earphone", "earphones", "earbud", "earbuds", "earpiece"]):
            return "audio"

        for family_name, family_info in self.families.items():
            if family_name == "audio":
                continue
            for k in family_info["keywords"]:
                if k == name_lower or f" {k} " in f" {name_lower} " or name_lower.startswith(f"{k} ") or name_lower.endswith(f" {k}"):
                    return family_name
                elif len(k) >= 4 and k in name_lower and not (k == "phone" and "headphone" in name_lower):
                    return family_name
        return None

    def disambiguate(
        self,
        image_bgr: np.ndarray,
        box: List[float],
        candidate_class: str,
        confidence: float
    ) -> Tuple[str, float, str]:
        """
        Applies morphological physics and fine-grained CLIP vision features to verify and disambiguate candidate detection.
        Returns: (verified_class, updated_confidence, scientific_rationale)
        """
        if not self.is_initialized:
            self.initialize()

        if not self.is_initialized or self.model is None:
            return candidate_class, confidence, "Baseline detection"

        family = self.find_family(candidate_class)
        if not family or family not in self.family_embeddings:
            return candidate_class, confidence, "Standard class signature"

        img_h, img_w = image_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # ---------------- 1. Morphological Physical Priors ----------------
        aspect_ratio = max(bw / bh, bh / bw)
        area_ratio = (bw * bh) / (img_w * img_h)
        is_vertical = bh >= bw

        # ---------------- 2. Region-of-Interest (ROI) Extraction with Margin ----------------
        pad_x = bw * 0.15
        pad_y = bh * 0.15
        cx1 = int(max(0, x1 - pad_x))
        cy1 = int(max(0, y1 - pad_y))
        cx2 = int(min(img_w, x2 + pad_x))
        cy2 = int(min(img_h, y2 + pad_y))

        crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
        if crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
            return candidate_class, confidence, "ROI too small for fine analysis"

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        try:
            crop_tensor = self.preprocess(crop_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(crop_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

            # Calculate cosine similarities for all classes in this family
            similarities: Dict[str, float] = {}
            for target_class, text_feat in self.family_embeddings[family].items():
                sim = float((img_feat @ text_feat.T)[0, 0].cpu().numpy())
                similarities[target_class] = sim

            # ---------------- 3. Multi-Physical Fusion Rules ----------------
            rationale_notes = []

            # Family-specific scientific heuristics:
            if family == "screens":
                # A modern smartphone held in hand typically has high aspect ratio (~2:1)
                # and occupies moderate or compact frame area compared to a room TV.
                if aspect_ratio >= 1.80:
                    similarities["smartphone"] = similarities.get("smartphone", 0) + 0.05
                    rationale_notes.append("Tall aspect ratio (~2:1) matches mobile screen geometry")
                if area_ratio < 0.35 and is_vertical:
                    similarities["smartphone"] = similarities.get("smartphone", 0) + 0.04
                    rationale_notes.append("Handheld vertical screen scale")
                elif area_ratio >= 0.35 and not is_vertical and aspect_ratio <= 1.85:
                    similarities["television"] = similarities.get("television", 0) + 0.06
                    rationale_notes.append("Wide 16:9 large-format television display area")

            elif family == "writing":
                # Marker vs Pen vs Pencil:
                # - Markers have lower length/thickness aspect ratio (thick body, barrel diameter > 12mm)
                # - Pens/Pencils are significantly thinner and longer
                thickness_ratio = min(bw, bh) / max(bw, bh)
                if thickness_ratio >= 0.18:
                    similarities["marker"] = similarities.get("marker", 0) + 0.04
                    similarities["highlighter"] = similarities.get("highlighter", 0) + 0.03
                    rationale_notes.append("Thick barrel profile (marker/highlighter morphology)")
                else:
                    similarities["pen"] = similarities.get("pen", 0) + 0.03
                    similarities["pencil"] = similarities.get("pencil", 0) + 0.03
                    rationale_notes.append("Slender cylindrical profile (pen/pencil morphology)")

            elif family == "audio":
                pass

            elif family == "containers":
                if aspect_ratio <= 1.6:
                    similarities["perfume bottle"] = similarities.get("perfume bottle", 0) + 0.04
                    rationale_notes.append("Compact flask proportions (perfume/cologne bottle)")
                else:
                    similarities["water bottle"] = similarities.get("water bottle", 0) + 0.04
                    rationale_notes.append("Elongated cylindrical vessel (water bottle/flask)")

            # Pick the top scoring class
            best_class = max(similarities, key=similarities.get)
            best_score = similarities[best_class]

            # Scale similarity to human-interpretable confidence (0.75 - 0.99)
            norm_conf = max(0.70, min(0.98, round(float(0.5 + (best_score * 1.5)), 2)))

            rationale_str = f"Scientific Dual-Stage Verification ({best_class.title()})"
            if rationale_notes:
                rationale_str += f": {rationale_notes[0]}"

            return best_class, norm_conf, rationale_str

        except Exception as e:
            print(f"[ScientificDiscriminator] Disambiguation error: {e}")
            return candidate_class, confidence, "Baseline detection"


# Global singleton instance
scientific_engine = ScientificDiscriminator()
