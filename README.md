# Real-Time Financial Fraud Detection Platform
Fraud scoring system built on the IEEE-CIS dataset (590K transactions, 3.5% fraud).
## Progress
- [x] EDA and feature engineering
- [x] Baseline XGBoost, time-based split (PR-AUC 0.4855)
- [x] MLflow experiment tracking
- [x] Improved model (PR-AUC 0.5073, ROC-AUC 0.8908)
- [ ] Feast feature store
- [ ] Kafka producer and FastAPI scoring service
- [ ] Graph neural network for fraud rings (PyTorch Geometric)
- [ ] Evidently drift monitoring and retraining trigger
- [ ] Analyst dashboard (Streamlit)
- [ ] Deployment (Docker Compose / Kubernetes)
## Key design decisions
- **Metric:** PR-AUC, not accuracy, because only 3.5% of transactions are fraud.
- **Split:** by time (80/20), because real fraud models predict the future.
- **No leakage:** user-level features use only past transactions.
## Repo structure
- `notebooks/` EDA, features, and model training
- `docs/` feature importance chart and experiment log
