# Yield prediction training

This script builds graph-based features from the companion plants network and trains two regressors (XGBoost and RandomForest) to predict the Help_Rank.

Outputs are saved under `artifacts/`:
- `yield_xgb_model.joblib` – XGBoost model
- `yield_rf_model.joblib` – RandomForest model
- `feature_columns.json` – input feature order for inference
- `metrics.json` – test-set metrics and metadata

Run (PowerShell, using your Python 3.11 venv):

- Upgrade tools and install deps (optional if already installed):
  - & "..\\.venv311\\Scripts\\python.exe" -m pip install --upgrade pip setuptools wheel
  - & "..\\.venv311\\Scripts\\python.exe" -m pip install pandas numpy networkx scikit-learn xgboost joblib matplotlib

- Train:
  - & "..\\.venv311\\Scripts\\python.exe" "train_yield_model.py"

Notes:
- The script automatically loads CSVs from `../companion_plants`.
- For inference later, use `feature_columns.json` to construct the input vector in the same column order.