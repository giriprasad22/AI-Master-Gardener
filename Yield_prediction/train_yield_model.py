"""
Train yield/helpfulness rank prediction models from companion plant graph features.

What it does:
- Loads companion graph CSVs from ../companion_plants
- Builds a NetworkX graph (help/avoid edges)
- Computes centrality and degree-based features
- Merges with final plant rankings to predict Help_Rank
- Trains XGBoost and RandomForest regressors
- Prints RMSE/MSE/MAE/R2 on test set
- Saves models and metadata for later inference

Run:
  E.g., python train_yield_model.py
Environment:
  Uses the active Python environment; ensure required packages are installed:
  pandas, numpy, networkx, scikit-learn, xgboost, joblib, matplotlib (optional)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib


# Make relative paths resolve from this script's folder
os.chdir(Path(__file__).parent)


@dataclass
class Metrics:
    rmse: float
    mse: float
    mae: float
    r2: float


def normalize_name(x):
    if pd.isna(x):
        return x
    return str(x).strip().lower()


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    cp_dir = root / "companion_plants"
    help_df = pd.read_csv(cp_dir / "help_network.csv")
    avoid_df = pd.read_csv(cp_dir / "avoid_network.csv")
    companion_df = pd.read_csv(cp_dir / "companion_plants.csv")
    companion_veg_df = pd.read_csv(cp_dir / "companion_plants_veg.csv")
    rankings_df = pd.read_csv(cp_dir / "final_plant_rankings.csv")

    # Normalize string columns
    for df in [help_df, avoid_df, companion_df, companion_veg_df]:
        for col in ["Source Node", "Destination Node", "Source", "Destination", "source", "dest"]:
            if col in df.columns:
                df[col] = df[col].apply(normalize_name)

    if "Plant" in rankings_df.columns:
        rankings_df["Plant"] = rankings_df["Plant"].apply(normalize_name)
    else:
        raise KeyError("final_plant_rankings.csv must contain a 'Plant' column")

    # Coerce numeric ranks
    for col in ["Help_Rank", "Avoid_Rank"]:
        if col in rankings_df.columns:
            rankings_df[col] = pd.to_numeric(rankings_df[col], errors="coerce")
        else:
            rankings_df[col] = np.nan

    return help_df, avoid_df, companion_df, companion_veg_df, rankings_df


def build_graph(help_df: pd.DataFrame, avoid_df: pd.DataFrame, companion_df: pd.DataFrame, companion_veg_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    def add_edges_from_df(df: pd.DataFrame, src_col: str, dst_col: str, relation: str):
        if src_col in df.columns and dst_col in df.columns:
            for _, r in df[[src_col, dst_col]].dropna().iterrows():
                s = r[src_col]
                d = r[dst_col]
                if s == "" or d == "":
                    continue
                G.add_edge(s, d, relation=relation)

    # Prioritize canonical column names; many csvs use "Source Node"/"Destination Node"
    add_edges_from_df(help_df, "Source Node", "Destination Node", "help")
    add_edges_from_df(avoid_df, "Source Node", "Destination Node", "avoid")
    add_edges_from_df(companion_df, "Source Node", "Destination Node", "help")
    add_edges_from_df(companion_veg_df, "Source Node", "Destination Node", "help")

    print(f"Graph built: nodes={len(G.nodes())}, edges={len(G.edges())}")
    return G


def compute_features(G: nx.Graph) -> pd.DataFrame:
    # Centralities (can be moderately expensive on large graphs but our graphs are small)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G)

    features: List[Dict[str, float]] = []
    for node in G.nodes():
        edges_list = list(G.edges(node, data=True))
        help_degree = sum(1 for _, _, d in edges_list if d.get("relation") == "help")
        avoid_degree = sum(1 for _, _, d in edges_list if d.get("relation") == "avoid")
        deg = G.degree(node)
        help_ratio = help_degree / deg if deg > 0 else 0.0
        avoid_ratio = avoid_degree / deg if deg > 0 else 0.0
        features.append({
            "plant": node,
            "degree": deg,
            "help_degree": help_degree,
            "avoid_degree": avoid_degree,
            "net_helpfulness": help_degree - avoid_degree,
            "help_ratio": help_ratio,
            "avoid_ratio": avoid_ratio,
            "betweenness": betweenness.get(node, 0.0),
            "closeness": closeness.get(node, 0.0),
            "clustering": clustering.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
        })
    features_df = pd.DataFrame(features)
    print("features_df preview:")
    print(features_df.head())
    return features_df


def prepare_dataset(features_df: pd.DataFrame, rankings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    merged_df = features_df.merge(rankings_df[["Plant", "Help_Rank", "Avoid_Rank"]],
                                  left_on="plant", right_on="Plant", how="inner")
    print("Merged rows:", merged_df.shape[0])
    print(merged_df[["plant", "Help_Rank", "Avoid_Rank"]].head())
    merged_df = merged_df.dropna(subset=["Help_Rank"]).reset_index(drop=True)
    print("After dropping missing Help_Rank:", merged_df.shape[0])

    feature_cols = [
        "degree", "help_degree", "avoid_degree", "net_helpfulness",
        "help_ratio", "avoid_ratio", "betweenness", "closeness",
        "clustering", "pagerank", "Avoid_Rank",
    ]

    merged_df["Avoid_Rank"] = merged_df["Avoid_Rank"].fillna(merged_df["Avoid_Rank"].median())
    merged_df[feature_cols] = merged_df[feature_cols].fillna(0)
    X = merged_df[feature_cols].copy()
    y = merged_df["Help_Rank"].copy()
    print("X shape:", X.shape, "y shape:", y.shape)
    print("Feature sample:\n", X.head())
    return X, y, feature_cols


def evaluate(model, X_t, y_t, name: str = "model") -> Metrics:
    yhat = model.predict(X_t)
    mse = mean_squared_error(y_t, yhat)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, yhat))
    mae = float(mean_absolute_error(y_t, yhat))
    print(f"{name}: RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return Metrics(rmse=rmse, mse=float(mse), mae=mae, r2=r2)


def main():
    help_df, avoid_df, companion_df, companion_veg_df, rankings_df = load_data()
    G = build_graph(help_df, avoid_df, companion_df, companion_veg_df)
    features_df = compute_features(G)
    X, y, feature_cols = prepare_dataset(features_df, rankings_df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

    # Models
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=0,  # let xgboost decide; avoids oversubscription on some setups
    )
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluate
    print("Evaluation on test set:")
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    rf_metrics = evaluate(rf_model, X_test, y_test, "RandomForest")

    # Save artifacts
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Models
    joblib.dump(xgb_model, out_dir / "yield_xgb_model.joblib")
    joblib.dump(rf_model, out_dir / "yield_rf_model.joblib")
    # Features and metadata
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": feature_cols,
        "xgb_metrics": asdict(xgb_metrics),
        "rf_metrics": asdict(rf_metrics),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved artifacts to:", out_dir.resolve())


if __name__ == "__main__":
    main()
