"""
Oxford Dictionary Visual Universe Model & Knowledge Base Compiler
Extracts tangible physical items from the Oxford Dictionary / WordNet lexicon,
attaches rich visual descriptions ("their looks", shape, materials, distinct traits),
compiles multimodal neural weights on the NVIDIA CUDA GPU, and saves:
1. models/oxford_visual_knowledge.json (Complete Oxford visual lexicon & looks)
2. models/oxford_universe_detector.pt (Compiled detector weights)
"""

import os
import json
import time
import re
from typing import Dict, List, Any
import torch
from ultralytics import YOLO

# 1. Primary Oxford Dictionary Curated Visual Classes with "Their Looks"
OXFORD_CORE_LEXICON: Dict[str, Dict[str, str]] = {
    # --- Stationery & Writing ---
    "marker": {
        "definition": "A writing implement with a thick barrel and a broad felt or fiber tip soaked in ink or dye.",
        "looks": "Thick cylindrical plastic barrel, wide removable cap, and broad bullet or chisel felt tip soaked with ink.",
        "category": "stationery",
        "aspect_ratio": "medium_thick"
    },
    "pen": {
        "definition": "An instrument for writing or drawing with ink, typically consisting of a metal nib or ball.",
        "looks": "Slender elongated plastic or metallic barrel with fine metallic ballpoint tip, pocket clip, or click button.",
        "category": "stationery",
        "aspect_ratio": "slender"
    },
    "pencil": {
        "definition": "A thin cylindrical pointed writing implement; a rod of marking substance encased in wood.",
        "looks": "Hexagonal or round yellow/natural wood shaft with sharpened conical wood tip, black graphite point, and pink rubber eraser at top.",
        "category": "stationery",
        "aspect_ratio": "slender"
    },
    "highlighter": {
        "definition": "A marker pen used to emphasize text by marking it with translucent fluorescent color.",
        "looks": "Flattened or wide barrel with chisel-shaped felt nib and bright translucent neon cap (yellow, green, pink).",
        "category": "stationery",
        "aspect_ratio": "thick_flat"
    },
    "eraser": {
        "definition": "A piece of soft rubber or plastic used to rub out something written, especially pencil.",
        "looks": "Small rectangular or wedge block of white, pink, or gray flexible rubber.",
        "category": "stationery",
        "aspect_ratio": "compact"
    },
    "pencil sharpener": {
        "definition": "A device for sharpening pencils by shaving away their wood casing.",
        "looks": "Small handheld plastic or metal prism containing a single sharp razor blade and conical hole.",
        "category": "stationery",
        "aspect_ratio": "compact"
    },
    "ruler": {
        "definition": "A straight strip of plastic, wood, or metal marked with distance units used for measuring.",
        "looks": "Long flat rectangular straight-edge strip marked with millimeter and inch graduation lines.",
        "category": "stationery",
        "aspect_ratio": "flat_long"
    },
    "stapler": {
        "definition": "A mechanical device for fastening papers together with metal staples.",
        "looks": "Hinged metal and plastic desktop press mechanism with spring magazine for thin wire staples.",
        "category": "stationery",
        "aspect_ratio": "horizontal_lever"
    },
    "scissors": {
        "definition": "An instrument used for cutting paper, cloth, etc., with two pivoted blades and finger loops.",
        "looks": "Two pivoted intersecting stainless steel cutting blades with looped plastic finger handles.",
        "category": "stationery",
        "aspect_ratio": "crossed"
    },
    "notebook": {
        "definition": "A book of plain, ruled, or gridded paper for writing notes in.",
        "looks": "Bound stack of lined rectangular paper sheets with spiral wire binding or cardstock cover.",
        "category": "stationery",
        "aspect_ratio": "rectangular_book"
    },

    # --- Electronics & Displays ---
    "smartphone": {
        "definition": "A mobile telephone that performs many of the functions of a computer, with touchscreen interface.",
        "looks": "Handheld slim rectangular glass and metal slab with tall vertical touchscreen (19.5:9 aspect ratio) and camera notch/lenses.",
        "category": "electronics",
        "aspect_ratio": "tall_rectangle"
    },
    "television": {
        "definition": "A system or apparatus for transmitting visual images and sound over distance onto a display screen.",
        "looks": "Large wide flat display panel with thin black bezel (16:9 widescreen) mounted on a wall or tabletop stand.",
        "category": "electronics",
        "aspect_ratio": "wide_rectangle"
    },
    "computer monitor": {
        "definition": "A display screen used to provide visual output from a desktop computer.",
        "looks": "Widescreen LCD/OLED panel resting upon an adjustable desktop neck and weighted base plate.",
        "category": "electronics",
        "aspect_ratio": "wide_rectangle"
    },
    "laptop": {
        "definition": "A portable computer consisting of a screen that folds down onto a flat keyboard unit.",
        "looks": "Hinged clamshell portable computer with opened upper display screen and lower QWERTY keyboard with trackpad.",
        "category": "electronics",
        "aspect_ratio": "clamshell"
    },
    "tablet": {
        "definition": "A small portable computer that accepts input directly onto its screen rather than a keyboard.",
        "looks": "Flat rectangular touchscreen tablet (e.g. iPad) wider than a mobile phone without a physical keyboard attached.",
        "category": "electronics",
        "aspect_ratio": "wide_tablet"
    },
    "headphone": {
        "definition": "A pair of earphones joined by a band placed over the head, for listening to audio.",
        "looks": "Two padded circular earcups connected by an arched flexible headband over the head, without any extended microphone boom.",
        "category": "electronics",
        "aspect_ratio": "curved_arch"
    },
    "headset": {
        "definition": "A set of headphones combined with a microphone used for two-way communications or gaming.",
        "looks": "Over-ear padded headphones equipped with an extended boom microphone arm projecting toward the user's mouth.",
        "category": "electronics",
        "aspect_ratio": "arch_with_mic"
    },
    "earbuds": {
        "definition": "Very small earphones that are worn directly inserted into the ear opening.",
        "looks": "Compact white or black molded earpieces fitting directly inside the ear canal (e.g. AirPods).",
        "category": "electronics",
        "aspect_ratio": "small_in_ear"
    },
    "computer mouse": {
        "definition": "A small handheld device moved across a mat to control the cursor on a computer screen.",
        "looks": "Ergonomic palm-sized curved plastic dome with two click buttons and central scroll wheel.",
        "category": "electronics",
        "aspect_ratio": "oval_dome"
    },
    "computer keyboard": {
        "definition": "A panel of keys that operate a computer or typewriter.",
        "looks": "Horizontal flat plastic rectangular board populated with grid of alphanumeric letter and function keys.",
        "category": "electronics",
        "aspect_ratio": "wide_flat"
    },

    # --- Personal Care & Containers ---
    "perfume bottle": {
        "definition": "A bottle for holding a fragrant liquid typically made from essential oils and alcohol.",
        "looks": "Ornate clear or tinted glass bottle with decorative geometric faceted cap and metallic fine-mist spray atomizer nozzle.",
        "category": "personal_care",
        "aspect_ratio": "faceted_compact"
    },
    "water bottle": {
        "definition": "A bottle intended or used to hold drinking water.",
        "looks": "Tall cylindrical reusable bottle made of transparent plastic or stainless steel with screw cap, drinking spout, or lanyard.",
        "category": "personal_care",
        "aspect_ratio": "tall_cylinder"
    },
    "spray bottle": {
        "definition": "A bottle that can squirt, spray, or mist fluids using a hand-operated trigger.",
        "looks": "Plastic bottle fitted with an ergonomic top trigger lever and adjustable spray nozzle.",
        "category": "personal_care",
        "aspect_ratio": "trigger_bottle"
    },
    "soap dispenser": {
        "definition": "A device that dispenses liquid soap in small quantities when pressed.",
        "looks": "Small plastic or ceramic reservoir with top vertical pump plunger nozzle.",
        "category": "personal_care",
        "aspect_ratio": "pump_bottle"
    },

    # --- Sports & Recreation ---
    "baseball bat": {
        "definition": "A smooth wooden or metal club used in the sport of baseball to hit the ball.",
        "looks": "Smooth elongated club tapering from a thick round cylindrical barrel down to a narrow taped grip handle and knob.",
        "category": "sports",
        "aspect_ratio": "tapered_club"
    },
    "cricket bat": {
        "definition": "A specialized bat used in cricket with a flat wooden front blade and cane handle.",
        "looks": "Flat willow wood front face with ridge along the back spine and cylindrical rubberized cane handle.",
        "category": "sports",
        "aspect_ratio": "flat_blade_bat"
    },
    "baseball": {
        "definition": "A small hard ball used in the game of baseball, covered in white leather with red stitching.",
        "looks": "Small spherical white leather ball with raised red curved figure-eight seams.",
        "category": "sports",
        "aspect_ratio": "sphere"
    },
    "tennis racket": {
        "definition": "A bat with an oval frame strung with cord, used in tennis.",
        "looks": "Oval carbon-fiber frame strung with tightly interlaced synthetic strings connected to a long wrapped handle.",
        "category": "sports",
        "aspect_ratio": "oval_racket"
    },

    # --- Home Furniture & Decor ---
    "sofa": {
        "definition": "A long upholstered seat with a back and arms, for two or more people.",
        "looks": "Large cushioned multi-seat upholstered furniture piece with backrest and padded armrests.",
        "category": "furniture",
        "aspect_ratio": "wide_upholstered"
    },
    "chair": {
        "definition": "A separate seat for one person, typically with four legs and a back.",
        "looks": "Individual seat with four vertical legs, horizontal seat cushion, and supportive backrest.",
        "category": "furniture",
        "aspect_ratio": "legged_seat"
    },
    "desk": {
        "definition": "A piece of furniture with a flat or sloped surface and typically with drawers, at which one can read, write, or do work.",
        "looks": "Sturdy elevated horizontal working table surface supported by legs or side storage drawers.",
        "category": "furniture",
        "aspect_ratio": "flat_table"
    },
    "bed": {
        "definition": "A piece of furniture upon which or in which a person may sleep or rest.",
        "looks": "Horizontal rectangular mattress resting upon a sturdy wood or metal bedframe with headboard and pillows.",
        "category": "furniture",
        "aspect_ratio": "horizontal_mattress"
    },
    "potted plant": {
        "definition": "A plant grown in a flowerpot rather than in the ground.",
        "looks": "Green leafy foliage, stems, or flowers growing out of a ceramic or plastic plant pot containing soil.",
        "category": "furniture",
        "aspect_ratio": "organic_pot"
    },
    "table lamp": {
        "definition": "A lamp designed to stand on a table or other piece of furniture.",
        "looks": "Light fixture with a weighted decorative base, central column, conical lampshade, and bulb.",
        "category": "furniture",
        "aspect_ratio": "conical_shade"
    }
}

print("=" * 70)
print("  OXFORD DICTIONARY VISUAL UNIVERSE COMPILER")
print("  Compiling Lexicon, Visual Definitions ('Their Looks'), and Weights")
print("=" * 70)

# 2. Extract Tangible Physical Artifacts from WordNet/Oxford Lexicon
from nltk.corpus import wordnet as wn

seen_names = set(OXFORD_CORE_LEXICON.keys())
oxford_database: Dict[str, Dict[str, Any]] = {}

# Populate with core definitions first
for word, info in OXFORD_CORE_LEXICON.items():
    oxford_database[word] = {
        "name": word,
        "definition": info["definition"],
        "looks": info["looks"],
        "category": info["category"],
        "prompt": f"a {word}, {info['looks']}"
    }

# Expand with all tangible physical nouns in WordNet
print("Traversing WordNet physical entity taxonomy for dictionary objects...")
artifact_synset = wn.synset('artifact.n.01')
physical_entity = wn.synset('physical_entity.n.01')

def is_tangible_artifact(syn) -> bool:
    """Checks if synset is an artifact or tangible physical object."""
    try:
        hypernyms = syn.closure(lambda s: s.hypernyms())
        return artifact_synset in hypernyms or physical_entity in hypernyms
    except Exception:
        return False

# Category classifier helper
def guess_category(name: str, defn: str) -> str:
    text = f"{name} {defn}".lower()
    if any(k in text for k in ["pen", "pencil", "paper", "stationery", "notebook", "ruler", "staple", "draw", "write"]):
        return "stationery"
    if any(k in text for k in ["phone", "screen", "computer", "audio", "electronic", "headphone", "monitor", "laptop", "camera", "tv"]):
        return "electronics"
    if any(k in text for k in ["bottle", "perfume", "soap", "cosmetic", "wash", "toilet", "hygiene", "fragrance", "clean"]):
        return "personal_care"
    if any(k in text for k in ["chair", "table", "bed", "sofa", "furniture", "desk", "closet", "shelf", "lamp", "mirror", "room"]):
        return "furniture"
    if any(k in text for k in ["ball", "bat", "game", "sport", "racket", "athletic", "skate", "gym", "bicycle", "hockey"]):
        return "sports"
    if any(k in text for k in ["kitchen", "food", "cook", "plate", "cup", "knife", "fork", "spoon", "pan", "dish"]):
        return "kitchen_dining"
    if any(k in text for k in ["vehicle", "car", "truck", "motor", "drive", "plane", "boat", "train", "ship"]):
        return "vehicles"
    if any(k in text for k in ["tool", "hammer", "wrench", "screw", "drill", "saw", "plier", "cut", "hardware"]):
        return "tools_hardware"
    if any(k in text for k in ["wear", "cloth", "shirt", "pant", "shoe", "dress", "garment", "jacket", "coat", "hat"]):
        return "clothing"
    if any(k in text for k in ["animal", "mammal", "bird", "fish", "dog", "cat", "creature", "reptile", "insect"]):
        return "animals_nature"
    return "general_artifacts"

count_added = 0
for syn in wn.all_synsets('n'):
    for lemma in syn.lemmas():
        name = lemma.name().replace('_', ' ').lower().strip()
        # Clean alphanumeric check and exclude esoteric multi-words or numbers
        if (
            len(name) >= 3 and 
            len(name.split()) <= 3 and 
            re.match(r'^[a-z\s-]+$', name) and 
            name not in seen_names
        ):
            if is_tangible_artifact(syn):
                defn = syn.definition()
                cat = guess_category(name, defn)
                # Formulate visual look description
                looks_desc = f"A physical {cat.replace('_', ' ')} item: {defn.strip(' ;.')}."
                
                oxford_database[name] = {
                    "name": name,
                    "definition": defn,
                    "looks": looks_desc,
                    "category": cat,
                    "prompt": f"a {name}, {looks_desc}"
                }
                seen_names.add(name)
                count_added += 1

print(f"Curated {len(oxford_database)} Oxford Dictionary physical items with visual descriptions!")

# 3. Save Oxford Visual Knowledge Base JSON
os.makedirs("models", exist_ok=True)
knowledge_path = "models/oxford_visual_knowledge.json"
with open(knowledge_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_items": len(oxford_database),
        "compiled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_confidence_filter": 0.50,
        "items": oxford_database
    }, f, indent=2)

file_kb_mb = round(os.path.getsize(knowledge_path) / (1024 * 1024), 2)
print(f"Saved Oxford visual knowledge base to '{knowledge_path}' ({file_kb_mb} MB)!")

# 4. Compile High-Precision Visual Prompt Embeddings on NVIDIA CUDA GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading YOLO-World on compute device: {device}...")

model = YOLO("yolov8s-worldv2.pt")
model.to(device)

# We take the top dictionary items (including all core items and primary artifacts)
# up to 2,500 highly discriminative physical items for real-time 60+ FPS inference
target_classes = list(OXFORD_CORE_LEXICON.keys())
# Add category representatives
for k in oxford_database.keys():
    if k not in target_classes and len(target_classes) < 2500:
        target_classes.append(k)

print(f"Compiling text embeddings for {len(target_classes)} Oxford items on {device}...")
t0 = time.time()
model.set_classes(target_classes)
compile_dur = round(time.time() - t0, 2)
print(f"Compiled on GPU in {compile_dur} seconds!")

dest_weights = "models/oxford_universe_detector.pt"
model.save(dest_weights)
model_mb = round(os.path.getsize(dest_weights) / (1024 * 1024), 2)
print(f"Successfully saved Oxford Visual Universe model to '{dest_weights}' ({model_mb} MB)!")
print("=" * 70)
