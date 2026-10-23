import os
import time
import yaml
import torch
from ultralytics import YOLO
from nltk.corpus import wordnet as wn

print("=" * 60)
print("🚀 Assembling 5,000-Class Universe Model on NVIDIA CUDA GPU")
print("=" * 60)

# 1. Target fine-grained classes explicitly requested by the user
PRIORITY_CLASSES = [
    # Distinct Audio & Communications
    "headphone",
    "headset",
    "gaming headset",
    "wireless headphones",
    "earphones",
    "earbuds",
    "airpods",
    "microphone",
    "boom microphone",

    # Distinct Perfumes & Fragrances
    "perfume",
    "perfume bottle",
    "cologne",
    "fragrance bottle",
    "body spray",
    "lotion bottle",
    "deodorant",

    # Distinct Water Bottles & Drinkware
    "water bottle",
    "plastic water bottle",
    "metal water bottle",
    "reusable water bottle",
    "sports bottle",
    "thermos",
    "flask",
    "tumbler",
    "glass bottle",
    "wine bottle",
    "beer bottle",
    "soda bottle",
    "spray bottle",

    # Distinct Stationery & Writing Implements
    "pen",
    "pencil",
    "ballpoint pen",
    "fountain pen",
    "gel pen",
    "wooden pencil",
    "mechanical pencil",
    "colored pencil",
    "marker",
    "highlighter",
    "whiteboard marker",
    "eraser",
    "pencil sharpener",
    "ruler",
    "measuring tape",
    "stapler",
    "staple remover",
    "paperclip",
    "binder clip",
    "notebook",
    "notepad",
    "spiral notebook",
    "binder",
    "folder",
    "sticky note",
    "pencil case",
    "calculator",
    "scissors",
    "glue stick",
    "scotch tape",
    "clipboard",
    "book",

    # Distinct Sports & Bats
    "baseball bat",
    "wooden baseball bat",
    "aluminum baseball bat",
    "baseball",
    "baseball glove",
    "softball bat",
    "cricket bat",
    "tennis racket",
    "badminton racket",
    "ping pong paddle",
    "hockey stick",
    "golf club",

    # Comprehensive Home Furniture & Architecture
    "sofa", "couch", "sectional sofa", "armchair", "recliner", "chair", "office chair", "gaming chair",
    "dining chair", "rocking chair", "bar stool", "bench", "bed", "queen bed", "king bed", "bunk bed",
    "crib", "mattress", "headboard", "pillow", "throw pillow", "cushion", "blanket", "comforter", "bedspread",
    "dining table", "coffee table", "end table", "bedside table", "nightstand", "desk", "computer desk",
    "wardrobe", "closet", "dresser", "chest of drawers", "bookshelf", "bookcase", "cabinet", "sideboard",
    "credenza", "ottoman", "futon", "table lamp", "floor lamp", "desk lamp", "lamp", "chandelier", "pendant light",
    "ceiling light", "mirror", "full length mirror", "wall mirror", "window", "sliding window", "door", "sliding door",
    "front door", "french door", "curtain", "drapes", "blinds", "carpet", "rug", "doormat", "picture frame",
    "photo frame", "painting", "wall art", "potted plant", "flowerpot", "vase", "fireplace", "trash can",
    "laundry basket", "towel rack", "shower curtain", "bathtub", "toilet", "bathroom sink", "kitchen sink",
    "faucet", "kitchen counter", "kitchen island",

    # Comprehensive Household Electronics & Appliances
    "laptop", "desktop computer", "computer monitor", "tablet computer", "smartphone", "cell phone",
    "smartwatch", "television", "remote control", "computer keyboard", "computer mouse", "trackpad",
    "printer", "scanner", "wifi router", "modem", "game console", "game controller", "gamepad", "vr headset",
    "webcam", "digital camera", "dslr camera", "camcorder", "projector", "microwave oven", "toaster",
    "toaster oven", "air fryer", "coffee maker", "espresso machine", "electric kettle", "blender",
    "food processor", "refrigerator", "freezer", "dishwasher", "washing machine", "clothes dryer",
    "vacuum cleaner", "robot vacuum", "iron", "ironing board", "hair dryer", "electric shaver",
    "electric toothbrush", "electric fan", "ceiling fan", "desk fan", "air conditioner", "space heater",
    "air purifier", "humidifier", "dehumidifier", "power strip", "extension cord", "charger", "power bank",
    "wall clock", "alarm clock"
]

# 2. Extract classes from Vision Benchmark datasets (LVIS, OpenImages, Objects365, ImageNet)
import ultralytics
cfg_dir = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "datasets")
benchmark_classes = []

for fname in ["lvis.yaml", "open-images-v7.yaml", "Objects365.yaml", "ImageNet.yaml"]:
    p = os.path.join(cfg_dir, fname)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            d = yaml.safe_load(f)
            names = d.get("names", {})
            items = list(names.values()) if isinstance(names, dict) else names
            for item in items:
                clean = item.split("/")[0].strip().replace("_", " ").lower()
                if clean and clean not in benchmark_classes:
                    benchmark_classes.append(clean)

print(f"Loaded {len(benchmark_classes)} benchmark classes.")

# 3. Extract physical artifacts from WordNet
artifact = wn.synset("artifact.n.01")
hyponyms = set([artifact])

def get_all_hyponyms(syn):
    for h in syn.hyponyms():
        if h not in hyponyms:
            hyponyms.add(h)
            get_all_hyponyms(h)

get_all_hyponyms(artifact)

wordnet_classes = []
for s in hyponyms:
    for lemma in s.lemmas():
        name = lemma.name().replace("_", " ").lower()
        if (
            len(name) >= 3
            and name.replace(" ", "").isalpha()
            and len(name.split()) <= 2
            and not any(bad in name for bad in ["genus", "family", "order", "suborder"])
        ):
            if name not in wordnet_classes:
                wordnet_classes.append(name)

print(f"Loaded {len(wordnet_classes)} WordNet physical artifact classes.")

# 4. Assemble the final exactly 5,000-class list
final_classes = []
seen = set()

def add_class(c):
    c_clean = c.strip().lower()
    if c_clean and c_clean not in seen and len(final_classes) < 5000:
        seen.add(c_clean)
        final_classes.append(c_clean)

# Priority first
for c in PRIORITY_CLASSES:
    add_class(c)

# Vision benchmarks second
for c in benchmark_classes:
    add_class(c)

# WordNet physical objects third
for c in wordnet_classes:
    add_class(c)

print(f"Total curated classes in 5,000 Universe: {len(final_classes)}")
assert len(final_classes) == 5000, f"Expected 5000 classes, got {len(final_classes)}"

# Check critical requested classes are present
for target in ["headphone", "headset", "perfume", "water bottle", "pen", "pencil", "baseball bat"]:
    assert target in seen, f"Missing target: {target}"
print("All target classes verified present and distinct!")

# 5. Load YOLO-World model onto CUDA GPU and bake the 5,000 classes
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compiling 5,000 class weights on device: {device}...")

t0 = time.time()
model = YOLO("yolov8s-worldv2.pt")
model.to(device)

# Set the 5000 classes
model.set_classes(final_classes)
print(f"GPU text embeddings computed in {round(time.time() - t0, 2)} seconds!")

dest_path = "models/universe_5000_mega_detector.pt"
model.save(dest_path)
file_size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
print(f"Successfully saved 5,000-class model to '{dest_path}' ({file_size_mb} MB)!")
