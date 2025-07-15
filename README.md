```bash
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
│   └── SC1_NetflixIndia_embeddings.pkl
│   └── SC1_NetflixIndia_kpi_features.json
│   └── SC1_NetflixIndia_meta_context.json
│
├── prompts/ 
│   └── base_prompts.txt (TBD)
│   └── generated_options.json
│
├── src/
│   └── embedding_pipeline.py
│   └── kpi_feature_engineering
│   └── meta_context_vector
│   └── strategy_agent.py
│   └── feedback_agent.py (TBD)
│
├── demo_app/ (TBD)
│   └── interface.py or streamlit_app.py
```