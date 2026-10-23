import os
import json
import time
import yaml
import torch
from ultralytics import YOLO
from nltk.corpus import wordnet as wn

print("=" * 70)
print("🚀 Compiling 10,000+ Object Universe Detector on NVIDIA CUDA GPU")
print("=" * 70)

# 1. High-Priority Fine-Grained Classes per Section
SECTION_TARGETS = {
    "stationery": [
        "pen", "pencil", "ballpoint pen", "fountain pen", "gel pen", "rollerball pen",
        "wooden pencil", "mechanical pencil", "colored pencil", "graphite pencil",
        "marker", "highlighter", "whiteboard marker", "permanent marker", "eraser",
        "pencil sharpener", "ruler", "measuring tape", "stapler", "staple remover",
        "paperclip", "binder clip", "notebook", "notepad", "spiral notebook", "diary",
        "binder", "folder", "manila folder", "sticky note", "post-it note", "pencil case",
        "calculator", "scientific calculator", "scissors", "glue stick", "liquid glue",
        "scotch tape", "masking tape", "duct tape", "clipboard", "book", "textbook",
        "index card", "correction tape", "hole puncher", "rubber band", "compass",
        "protractor", "drawing paper", "sketchbook", "envelope", "chalk", "whiteboard"
    ],
    "electronics": [
        "laptop", "desktop computer", "computer monitor", "tablet computer", "smartphone",
        "cell phone", "headphone", "headset", "gaming headset", "wireless headphones",
        "over-ear headphones", "earphones", "earbuds", "airpods", "boom microphone",
        "microphone", "smartwatch", "fitness tracker", "television", "remote control",
        "computer keyboard", "mechanical keyboard", "computer mouse", "wireless mouse",
        "trackpad", "printer", "laser printer", "scanner", "wifi router", "modem",
        "game console", "game controller", "gamepad", "joystick", "vr headset", "webcam",
        "digital camera", "dslr camera", "camcorder", "projector", "microwave oven",
        "toaster", "toaster oven", "air fryer", "coffee maker", "espresso machine",
        "electric kettle", "blender", "food processor", "refrigerator", "freezer",
        "dishwasher", "washing machine", "clothes dryer", "vacuum cleaner", "robot vacuum",
        "iron", "hair dryer", "electric shaver", "electric toothbrush", "electric fan",
        "ceiling fan", "desk fan", "air conditioner", "space heater", "air purifier",
        "humidifier", "dehumidifier", "power strip", "extension cord", "phone charger",
        "power bank", "usb flash drive", "external hard drive", "audio speaker", "subwoofer",
        "amplifier", "soundbar", "alarm clock", "wall clock"
    ],
    "furniture": [
        "sofa", "couch", "sectional sofa", "loveseat", "armchair", "recliner", "chair",
        "office chair", "gaming chair", "dining chair", "rocking chair", "bar stool",
        "folding chair", "bench", "park bench", "bed", "queen bed", "king bed", "single bed",
        "bunk bed", "canopy bed", "crib", "mattress", "headboard", "pillow", "throw pillow",
        "cushion", "blanket", "comforter", "duvet", "bedspread", "dining table",
        "coffee table", "end table", "bedside table", "nightstand", "desk", "computer desk",
        "writing desk", "standing desk", "wardrobe", "closet", "dresser", "chest of drawers",
        "bookshelf", "bookcase", "cabinet", "filing cabinet", "sideboard", "credenza",
        "hutch", "armoire", "ottoman", "futon", "beanbag", "table lamp", "floor lamp",
        "desk lamp", "lamp", "chandelier", "pendant light", "ceiling light", "mirror",
        "full length mirror", "wall mirror", "window", "sliding window", "bay window",
        "door", "sliding door", "front door", "french door", "curtain", "drapes", "blinds",
        "carpet", "rug", "area rug", "doormat", "picture frame", "photo frame", "painting",
        "wall art", "potted plant", "flowerpot", "vase", "fireplace", "trash can",
        "wastebasket", "laundry basket", "towel rack", "shower curtain", "bathtub",
        "toilet", "bathroom sink", "kitchen sink", "faucet", "kitchen counter", "kitchen island"
    ],
    "personal_care": [
        "perfume", "perfume bottle", "cologne", "cologne bottle", "fragrance spray",
        "body spray", "eau de parfum", "water bottle", "plastic water bottle",
        "metal water bottle", "reusable water bottle", "sports bottle", "thermos",
        "flask", "tumbler", "drinking bottle", "glass bottle", "spray bottle",
        "shampoo bottle", "conditioner bottle", "lotion bottle", "hand sanitizer",
        "liquid soap dispenser", "bar soap", "soap dish", "toothbrush", "toothpaste tube",
        "mouthwash bottle", "dental floss", "hairbrush", "comb", "hair roller", "hair clip",
        "nail clipper", "nail file", "tweezers", "razor", "shaving cream", "sunscreen bottle",
        "face cream jar", "lip balm", "lipstick", "mascara", "eyeliner", "makeup brush",
        "compact mirror", "sponge", "cotton swab", "cotton pad", "deodorant", "body wash",
        "bath towel", "hand towel", "washcloth"
    ],
    "sports": [
        "baseball bat", "wooden baseball bat", "aluminum baseball bat", "baseball",
        "baseball glove", "softball bat", "softball", "cricket bat", "cricket ball",
        "tennis racket", "tennis ball", "badminton racket", "shuttlecock", "ping pong paddle",
        "ping pong ball", "table tennis table", "golf club", "golf ball", "golf bag",
        "basketball", "basketball hoop", "soccer ball", "football", "rugby ball",
        "volleyball", "volleyball net", "bowling ball", "bowling pin", "hockey stick",
        "hockey puck", "ice skates", "roller skates", "rollerblades", "skateboard",
        "longboard", "surfboard", "snowboard", "skis", "ski poles", "bicycle",
        "mountain bike", "road bike", "helmet", "bicycle helmet", "protective goggles",
        "boxing gloves", "punching bag", "dumbbell", "barbell", "weight plate",
        "kettlebell", "jump rope", "yoga mat", "treadmill", "stationary bike",
        "fishing rod", "fishing reel", "tackle box", "canoe", "kayak", "paddle"
    ],
    "kitchen_dining": [
        "plate", "dinner plate", "salad plate", "bowl", "soup bowl", "cereal bowl",
        "cup", "coffee mug", "teacup", "drinking glass", "wine glass", "beer glass",
        "champagne flute", "shot glass", "pitcher", "carafe", "teapot", "coffee pot",
        "fork", "dinner fork", "salad fork", "knife", "butter knife", "steak knife",
        "chef knife", "bread knife", "paring knife", "spoon", "soup spoon", "teaspoon",
        "tablespoon", "chopsticks", "straw", "napkin", "placemat", "tablecloth",
        "frying pan", "saucepan", "stockpot", "wok", "casserole dish", "baking sheet",
        "muffin tin", "cutting board", "colander", "strainer", "measuring cup",
        "measuring spoons", "whisk", "spatula", "tongs", "ladle", "slotted spoon",
        "potato peeler", "can opener", "bottle opener", "corkscrew", "grater",
        "garlic press", "rolling pin", "oven mitt", "apron", "dish rack", "sponge"
    ],
    "vehicles": [
        "car", "sedan", "coupe", "convertible", "suv", "hatchback", "pickup truck",
        "truck", "semi truck", "van", "minivan", "bus", "school bus", "motorcycle",
        "scooter", "moped", "bicycle", "electric bike", "skateboard", "ambulance",
        "police car", "fire truck", "taxi", "garbage truck", "tractor", "bulldozer",
        "forklift", "crane", "airplane", "jet", "helicopter", "hot air balloon",
        "train", "locomotive", "subway train", "tram", "boat", "speedboat", "sailboat",
        "yacht", "ferry", "canoe", "kayak", "cruise ship", "cargo ship", "submarine"
    ],
    "tools_hardware": [
        "hammer", "claw hammer", "sledgehammer", "mallet", "screwdriver",
        "phillips screwdriver", "flathead screwdriver", "wrench", "adjustable wrench",
        "pipe wrench", "pliers", "needle nose pliers", "wire cutters", "wire stripper",
        "saw", "handsaw", "hacksaw", "circular saw", "jigsaw", "chainsaw", "power drill",
        "cordless drill", "drill bit", "tape measure", "level", "torpedo level", "utility knife",
        "box cutter", "chisel", "file", "sandpaper", "clamp", "vise", "ladder", "step stool",
        "toolbox", "tool belt", "flashlight", "headlamp", "extension cord", "soldering iron",
        "heat gun", "nail", "screw", "bolt", "nut", "washer", "anchor", "staple gun"
    ],
    "clothing": [
        "t-shirt", "shirt", "polo shirt", "blouse", "sweater", "hoodie", "sweatshirt",
        "jacket", "leather jacket", "denim jacket", "coat", "winter coat", "raincoat",
        "suit", "blazer", "vest", "pants", "jeans", "trousers", "sweatpants", "shorts",
        "skirt", "dress", "evening gown", "bathrobe", "pajamas", "socks", "shoes",
        "sneakers", "running shoes", "dress shoes", "boots", "ankle boots", "hiking boots",
        "sandals", "flip-flops", "slippers", "high heels", "belt", "necktie", "bow tie",
        "scarf", "gloves", "mittens", "hat", "baseball cap", "beanie", "sun hat",
        "fedora", "glasses", "sunglasses", "wristwatch", "necklace", "bracelet", "ring",
        "earrings", "backpack", "handbag", "purse", "tote bag", "wallet", "suitcase"
    ],
    "animals_nature": [
        "dog", "puppy", "cat", "kitten", "bird", "parrot", "canary", "eagle", "hawk",
        "owl", "sparrow", "pigeon", "crow", "duck", "goose", "swan", "chicken", "rooster",
        "turkey", "penguin", "flamingo", "fish", "goldfish", "tropical fish", "shark",
        "whale", "dolphin", "turtle", "sea turtle", "frog", "toad", "lizard", "gecko",
        "snake", "chameleon", "alligator", "crocodile", "rabbit", "bunny", "hamster",
        "guinea pig", "mouse", "rat", "squirrel", "chipmunk", "hedgehog", "horse",
        "pony", "donkey", "mule", "cow", "calf", "bull", "ox", "sheep", "lamb", "goat",
        "pig", "piglet", "deer", "stag", "elk", "moose", "bear", "grizzly bear", "polar bear",
        "panda", "koala", "kangaroo", "lion", "tiger", "leopard", "cheetah", "jaguar",
        "wolf", "fox", "coyote", "hyena", "elephant", "rhino", "hippopotamus", "giraffe",
        "zebra", "camel", "llama", "alpaca", "monkey", "chimpanzee", "gorilla", "baboon",
        "butterfly", "moth", "bee", "honeybee", "wasp", "ant", "beetle", "ladybug",
        "dragonfly", "grasshopper", "spider", "snail", "crab", "lobster", "octopus"
    ]
}

# 2. Extract classes from Vision Benchmark Datasets (LVIS, OpenImages V7, Objects365, ImageNet)
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
                if (clean and clean not in benchmark_classes 
                    and not any(bad in clean for bad in ["genus", "family", "order", "suborder", "sp.", "diet"])):
                    benchmark_classes.append(clean)

print(f"Loaded {len(benchmark_classes)} clean classes from official vision datasets.")

# 3. Extract Concrete Physical Artifacts from WordNet
concrete_roots = [
    "instrumentality.n.03", "furnishing.n.01", "container.n.01",
    "consumer_goods.n.01", "garment.n.01", "food.n.01", 
    "conveyance.n.01", "organism.n.01"
]
hyponym_syns = set()
for r in concrete_roots:
    try:
        root_syn = wn.synset(r)
        def recurse(s):
            for h in s.hyponyms():
                if h not in hyponym_syns:
                    hyponym_syns.add(h)
                    recurse(h)
        recurse(root_syn)
    except Exception as e:
        print(f"Error traversing {r}: {e}")

wordnet_classes = []
for s in hyponym_syns:
    for l in s.lemmas():
        name = l.name().replace("_", " ").lower()
        if (
            len(name) >= 3
            and name.replace(" ", "").isalpha()
            and len(name.split()) <= 2
            and not any(bad in name for bad in ["genus", "family", "order", "suborder", "diet", "future", "whore", "slave"])
        ):
            if name not in wordnet_classes:
                wordnet_classes.append(name)

print(f"Extracted {len(wordnet_classes)} clean physical nouns from WordNet.")

# 4. Assemble Exactly 10,000 Unique Physical Object Classes
final_classes = []
seen = set()

def add_class(c):
    clean = c.strip().lower()
    if clean and clean not in seen and len(final_classes) < 10000:
        seen.add(clean)
        final_classes.append(clean)

# Add all priority section items first
for section, items in SECTION_TARGETS.items():
    for item in items:
        add_class(item)

# Add vision benchmarks second
for item in benchmark_classes:
    add_class(item)

# Add clean WordNet physical objects until 10,000 reached
for item in wordnet_classes:
    add_class(item)

print(f"Total curated classes in 10,000-Universe: {len(final_classes)}")
assert len(final_classes) == 10000, f"Expected 10000 classes, got {len(final_classes)}"

# 5. Build Section Mapping for UI Filtering & Session Switcher
# Create map of section -> list of class names or class indices
taxonomy = {
    "sections": {
        "all": {
            "name": "🌐 All-in-One Universe (10,000+ Objects)",
            "count": len(final_classes),
            "classes": []  # empty signifies all
        }
    }
}

# Auto-classify words into sections for fast UI filtering
for sec_key, sec_targets in SECTION_TARGETS.items():
    sec_names = []
    # Include exact targets
    for t in sec_targets:
        if t in seen:
            sec_names.append(t)
    
    # Also add matching words from 10,000 list based on keywords
    keywords = {
        "stationery": ["pen", "pencil", "paper", "book", "note", "ruler", "staple", "tape", "folder", "binder", "envelope", "chalk", "card", "desk", "file"],
        "electronics": ["laptop", "phone", "monitor", "headphone", "headset", "ear", "camera", "tv", "audio", "mic", "electric", "charger", "cord", "battery", "radio", "clock"],
        "furniture": ["sofa", "couch", "chair", "table", "bed", "lamp", "closet", "cabinet", "shelf", "mirror", "window", "door", "curtain", "rug", "frame", "plant", "pillow", "blanket", "drawer"],
        "personal_care": ["perfume", "cologne", "bottle", "soap", "cream", "lotion", "razor", "brush", "comb", "towel", "spray", "sanitizer", "wash", "balm"],
        "sports": ["bat", "ball", "racket", "club", "skate", "surf", "ski", "glove", "helmet", "bike", "bicycle", "board", "paddle", "fitness", "dumbbell"],
        "kitchen_dining": ["plate", "bowl", "cup", "glass", "fork", "knife", "spoon", "pan", "pot", "dish", "cook", "kitchen", "food", "cake", "bread", "fruit"],
        "vehicles": ["car", "truck", "bus", "train", "plane", "airplane", "boat", "ship", "motorcycle", "bike", "vehicle", "van", "wagon"],
        "tools_hardware": ["hammer", "saw", "wrench", "plier", "drill", "tool", "nail", "screw", "bolt", "cutter", "level", "chisel"],
        "clothing": ["shirt", "pant", "shoe", "dress", "jacket", "coat", "hat", "cap", "sock", "glove", "boot", "belt", "tie", "suit", "skirt"],
        "animals_nature": ["dog", "cat", "bird", "fish", "tree", "flower", "animal", "bear", "horse", "cow", "lion", "tiger", "deer", "snake", "insect"]
    }
    
    kw_list = keywords.get(sec_key, [])
    for c in final_classes:
        if c not in sec_names and any(k in c for k in kw_list):
            sec_names.append(c)
            
    section_labels = {
        "stationery": "✏️ Stationery & Writing",
        "electronics": "💻 Electronics & Tech",
        "furniture": "🛋️ Home Furniture & Decor",
        "personal_care": "🧴 Perfumes, Bottles & Care",
        "sports": "⚾ Sports & Recreation",
        "kitchen_dining": "🍽️ Kitchen & Dining",
        "vehicles": "🚗 Vehicles & Automotive",
        "tools_hardware": "🛠️ Tools & Hardware",
        "clothing": "👕 Clothing & Apparel",
        "animals_nature": "🐕 Animals & Nature"
    }
    
    taxonomy["sections"][sec_key] = {
        "name": section_labels.get(sec_key, sec_key.title()),
        "count": len(sec_names),
        "classes": sec_names
    }

# Save taxonomy metadata JSON for the backend and frontend
taxonomy_path = "models/universe_10000_taxonomy.json"
with open(taxonomy_path, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2)
print(f"Saved taxonomy metadata to '{taxonomy_path}' with {len(taxonomy['sections'])} sections!")

# 6. Load YOLO-World onto NVIDIA CUDA GPU and Compile the 10,000 Classes
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Compiling 10,000-class weights on compute device: {device}...")

t0 = time.time()
model = YOLO("yolov8s-worldv2.pt")
model.to(device)

model.set_classes(final_classes)
print(f"GPU text embeddings computed in {round(time.time() - t0, 2)} seconds!")

dest_path = "models/universe_10000_mega_detector.pt"
model.save(dest_path)
file_size_mb = round(os.path.getsize(dest_path) / (1024 * 1024), 2)
print(f"Successfully saved 10,000-class model to '{dest_path}' ({file_size_mb} MB)!")
print("=" * 70)
