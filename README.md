project_root/
│
├── data/
│   └── usecases/
│       └── SC1_NetflixIndia/
│           ├── case_description.txt
│           ├── market_trends.txt
│           ├── internal_reports.csv
│           ├── stakeholder_notes.txt
│           └── decision_framework.json
│
├── embeddings/
│   └── SC1_NetflixIndia_embeddings.pkl  ← (optional, post-Step 3)
│
├── prompts/
│   └── base_prompts.txt
│   └── generated_options.json
│
├── src/
│   └── embedding_pipeline.py
│   └── strategy_agent.py
│   └── feedback_agent.py
│
├── demo_app/ (optional)
│   └── interface.py or streamlit_app.py
