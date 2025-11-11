# Step 3.3: Meta-Context Vector Engineering
# Save this file as: /src/meta_context_vector.py

import json
from pathlib import Path

# === Configuration ===
USECASE_ID = "SC1_NetflixIndia"
INPUT_PATH = Path("data/usecases") / USECASE_ID / "decision_framework.json"
OUTPUT_PATH = Path("embeddings") / f"{USECASE_ID}_meta_context.json"

# === Load Metadata Tags ===
with open(INPUT_PATH, "r") as f:
    decision_data = json.load(f)
tags = decision_data.get("metadata_tags", {})

# === Define all possible categories (for one-hot-like encoding)
possible_values = {
    "region": ["India", "US"],
    "industry": ["Streaming", "Finance", "Retail"],
    "urgency": ["low", "medium", "high"],
    "risk_tolerance": ["low", "medium", "high"],
    "hierarchy_level": ["CXO", "Middle_Manager", "Team_Lead"]
}

# === Generate Vector ===
context_vector = {}
for tag, options in possible_values.items():
    value = tags.get(tag)
    for option in options:
        key = f"{tag}_{option}"
        context_vector[key] = 1 if value == option else 0

# === Save Output ===
with open(OUTPUT_PATH, "w") as f:
    json.dump({USECASE_ID: context_vector}, f, indent=4)

print(f"✅ Meta-context vector saved to: {OUTPUT_PATH}")
