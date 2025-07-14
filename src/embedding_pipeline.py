# Step 3.1: Text Embedding Pipeline using SBERT
# Save this script as: /src/embedding_pipeline.py

import os
import re
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === Configuration ===
USECASE_ID = "SC1_NetflixIndia"
BASE_DIR = Path("data/usecases") / USECASE_ID
OUTPUT_PATH = BASE_DIR / "embeddings_text.pkl"
SEGMENTED_FILES = ["market_trends_segmented.txt", "stakeholder_notes_segmented.txt"]
UNSEGMENTED_FILES = {"case_description.txt": "CASE_DESCRIPTION"}

# === Load SBERT Model ===
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight but powerful

# === Embedding Dictionary ===
embedding_dict = {}

# === Process Unsegmented Files ===
for file_name, key in UNSEGMENTED_FILES.items():
    path = BASE_DIR / file_name
    if path.exists():
        text = path.read_text()
        embedding = model.encode(text.strip())
        embedding_dict[key] = embedding

# === Process Segmented Files ===
for file_name in SEGMENTED_FILES:
    path = BASE_DIR / file_name
    if not path.exists():
        continue
    with open(path, "r") as f:
        content = f.read()
    blocks = re.split(r"^##\s+", content, flags=re.MULTILINE)
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n", 1)
        if len(lines) == 2:
            section, body = lines
            section = section.strip().upper().replace(" ", "_")
            embedding = model.encode(body.strip())
            embedding_dict[section] = embedding

# === Save Output ===
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embedding_dict, f)

print(f"✅ Embeddings saved to: {OUTPUT_PATH}")
