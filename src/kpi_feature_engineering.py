# Step 3.2: Time-Series KPI Feature Engineering
# Save this file as: /src/kpi_feature_engineering.py

import pandas as pd
import json
from pathlib import Path

# === Configuration ===
USECASE_ID = "SC1_NetflixIndia"
DATA_PATH = Path("data/usecases") / USECASE_ID / "internal_reports.csv"
OUTPUT_PATH = Path("embeddings") / f"{USECASE_ID}_kpi_features.json"

# === Load Data ===
df = pd.read_csv(DATA_PATH)

# === Split by Region ===
india_df = df[df["Region"] == "India"].reset_index(drop=True)
us_df = df[df["Region"] == "US"].reset_index(drop=True)

# === Check Matching Quarters ===
assert all(india_df["Quarter"] == us_df["Quarter"]), "Mismatch in quarters!"

# === Feature Engineering ===
features = {}
for i in range(len(india_df)):
    quarter = india_df.loc[i, "Quarter"]
    india_row = india_df.loc[i]
    us_row = us_df.loc[i]

    features[quarter] = {
        "ARPU_ratio": round(india_row["ARPU"] / us_row["ARPU"], 3),
        "Churn_gap": round(india_row["Churn"] - us_row["Churn"], 3),
        "ServerCost_ratio": round(india_row["Server_Cost"] / us_row["Server_Cost"], 3),
        "Localization_gap": round(india_row["Localization"] - us_row["Localization"], 1),
        "WatchHour_delta": round(india_row["Avg_Watch_Hrs"] - us_row["Avg_Watch_Hrs"], 2)
    }

# === Save Output ===
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump({USECASE_ID: features}, f, indent=4)

print(f"✅ KPI features saved to: {OUTPUT_PATH}")
