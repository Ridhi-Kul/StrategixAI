# Step 4: Strategy Agent (LLM-based) for low-VRAM GPU (GTX 1650)
# Save as: /src/strategy_agent.py

import json
import pickle
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === Config ===
USECASE_ID = "SC1_NetflixIndia"
DATA_DIR = Path("data/usecases") / USECASE_ID
EMB_DIR = Path("embeddings")

# === Load Inputs ===
def load_text_embeddings():
    with open(EMB_DIR / f"{USECASE_ID}_embeddings.pkl", "rb") as f:
        return pickle.load(f)

def load_kpi_features():
    with open(EMB_DIR / f"{USECASE_ID}_kpi_features.json") as f:
        return json.load(f)[USECASE_ID]

def load_meta_context():
    with open(EMB_DIR / f"{USECASE_ID}_meta_context.json") as f:
        return json.load(f)[USECASE_ID]

def load_decision_framework():
    with open(DATA_DIR / "decision_framework.json") as f:
        return json.load(f)

# === Prompt Builder ===
def build_prompt(embeddings, kpis, meta, decision_info):
    quarter = list(kpis.keys())[-1]
    kpi_values = kpis[quarter]

    prompt = f"""You are a strategic advisor to Netflix India leadership.

The company is considering the following strategic question:
"{decision_info['decision_question']}"

Here is a summary of the most recent internal conditions:
- Watch hours trend (India vs US): {kpi_values['WatchHour_delta']}
- ARPU ratio (India/US): {kpi_values['ARPU_ratio']}
- Churn gap (India - US): {kpi_values['Churn_gap']}
- Server cost ratio: {kpi_values['ServerCost_ratio']}
- Localization gap: {kpi_values['Localization_gap']}

Contextual Tags:
{', '.join([k for k, v in meta.items() if v == 1])}

Based on the risks, opportunities, and stakeholder views:
Suggest 2–3 strategic options for this scenario, and explain the rationale behind each.

Strategic Options:
"""
    return prompt

# === Load Optimized LLM ===
def load_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=400)

# === Run Agent ===
def main():
    print("🚀 Loading inputs...")
    text_embeds = load_text_embeddings()
    kpis = load_kpi_features()
    meta = load_meta_context()
    decision_info = load_decision_framework()

    print("🧠 Building prompt...")
    prompt = build_prompt(text_embeds, kpis, meta, decision_info)

    print("🤖 Running LLM (TinyLlama float16)...")
    llm = load_llm()
    full_output = llm(prompt)[0]['generated_text']
    trimmed_output = full_output[len(prompt):].strip()

    print("\n📝 LLM Response Preview:\n" + "-"*40)
    print(trimmed_output[:800] + ("..." if len(trimmed_output) > 800 else ""))
    print("-" * 40)

    output_path = Path("prompts") / "generated_options.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "response": trimmed_output
        }, f, indent=4)

    print(f"✅ Output saved to: {output_path}")

if __name__ == "__main__":
    main()
