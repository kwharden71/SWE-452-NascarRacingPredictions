# ==========================================================
# NASCAR LGBMRanker — GridSearchCV Hyperparameter Tuning
#
# LGBMRanker requires group arrays alongside X/y, so it is
# not directly compatible with sklearn's GridSearchCV.
# This script implements an equivalent group-aware CV loop
# that scores every param combination via NDCG and then
# re-trains the best model on the full training set.
# ==========================================================

import pandas as pd
import numpy as np
import pickle
import os
import itertools
import warnings
from lightgbm import LGBMRanker
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")


# ----------------------------------------------------------
# 1.  Data preparation
# ----------------------------------------------------------

def load_and_prepare(path: str):
    data = pd.read_csv(path)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    data["Rank"] = data.groupby("Year")["Points"].rank(
        ascending=True, method="dense"
    )
    data["Rank"] = (data["Rank"] - 1).astype(int)
    data = data.sort_values(by="Year").reset_index(drop=True)
    return data


def split_data(data):
    years = data["Year"].unique()
    train_years, test_years = train_test_split(
        years, test_size=0.2, random_state=42
    )
    train_df = data[data["Year"].isin(train_years)].copy()
    test_df = data[data["Year"].isin(test_years)].copy()
    return train_df, test_df


COLUMNS_TO_DROP = ["id", "Driver", "Points", "Year", "Rank"]
MAX_RANK = 74   # label_gain list length must exceed max rank value (73)


# ----------------------------------------------------------
# 2.  Metrics
# ----------------------------------------------------------

def calculate_ndcg(group):
    """Per-year NDCG (higher is better, winner = most relevant)."""
    if len(group) < 2:
        return np.nan
    true_rel = (group["Rank"].max() - group["Rank"]).values
    y_true = [true_rel]
    y_score = [group["pred_score"].values]
    return ndcg_score(y_true, y_score)


def eval_model(model, df):
    """Return mean NDCG across all years in df."""
    X = df.drop(columns=COLUMNS_TO_DROP)
    df = df.copy()
    df["pred_score"] = model.predict(X)
    scores = df.groupby("Year").apply(calculate_ndcg, include_groups=False)
    return scores.mean()


# ----------------------------------------------------------
# 3.  Group-aware cross-validation for a single param set
# ----------------------------------------------------------

def group_kfold_cv(params: dict, train_df: pd.DataFrame, n_splits: int = 4) -> float:
    """
    Split training years into n_splits folds, train on fold train years,
    evaluate on fold val years.  Returns mean NDCG across folds.
    """
    years = train_df["Year"].unique()

    # Not enough groups for n_splits → reduce gracefully
    n_splits = min(n_splits, len(years))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold_train_idx, fold_val_idx in kf.split(years):
        fold_train_years = years[fold_train_idx]
        fold_val_years   = years[fold_val_idx]

        fold_train = train_df[train_df["Year"].isin(fold_train_years)]
        fold_val = train_df[train_df["Year"].isin(fold_val_years)].copy()

        q_fold_train = fold_train.groupby("Year").size().to_list()

        X_ft = fold_train.drop(columns=COLUMNS_TO_DROP)
        y_ft = fold_train["Rank"]

        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[i for i in range(MAX_RANK)],
            verbose=-1,
            **params,
        )
        model.fit(X_ft, y_ft, group=q_fold_train)

        score = eval_model(model, fold_val)
        fold_scores.append(score)

    return float(np.mean(fold_scores))


# ----------------------------------------------------------
# 4.  Hyperparameter grid
# ----------------------------------------------------------

PARAM_GRID = {
    # Number of boosting rounds
    "n_estimators":  [100, 300, 500],

    # Controls overfitting — smaller = more regularisation
    "learning_rate": [0.01, 0.05, 0.1],

    # Tree complexity
    "num_leaves":    [15, 31, 63],

    # Minimum samples in a leaf
    "min_child_samples": [5, 10, 20],

    # Column sub-sampling per tree
    "colsample_bytree": [0.8, 1.0],

    # Row sub-sampling per tree
    "subsample":     [0.8, 1.0],
}


# ----------------------------------------------------------
# 5.  Grid search loop
# ----------------------------------------------------------

def run_grid_search(train_df: pd.DataFrame, n_cv_splits: int = 4):
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    total  = len(combos)
    print(f"\n{'='*60}")
    print(f"  Grid Search: {total} combinations × {n_cv_splits}-fold CV")
    print(f"{'='*60}\n")

    best_score  = -np.inf
    best_params = None
    results     = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        score  = group_kfold_cv(params, train_df, n_splits=n_cv_splits)
        results.append({**params, "cv_ndcg": score})

        status = f"[{i:>3}/{total}]  NDCG={score:.4f}  {params}"
        print(status)

        if score > best_score:
            best_score  = score
            best_params = params
            print(f"  *** New best! NDCG={best_score:.4f} ***")

    results_df = pd.DataFrame(results).sort_values("cv_ndcg", ascending=False)
    return best_params, best_score, results_df


# ----------------------------------------------------------
# 6.  Final training & evaluation
# ----------------------------------------------------------

def train_final_model(best_params: dict, train_df: pd.DataFrame):
    q_train = train_df.groupby("Year").size().to_list()
    X_train = train_df.drop(columns=COLUMNS_TO_DROP)
    y_train = train_df["Rank"]

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[i for i in range(MAX_RANK)],
        verbose=-1,
        **best_params,
    )
    model.fit(X_train, y_train, group=q_train)
    return model


# ----------------------------------------------------------
# 7.  Main
# ----------------------------------------------------------

def Main():
    data = load_and_prepare("../data/nascar_driver_statistics.csv")
    train_df, test_df = split_data(data)

    # ── Grid search ──────────────────────────────────────
    best_params, best_cv_score, results_df = run_grid_search(train_df)

    print(f"\n{'='*60}")
    print("  Grid Search Complete")
    print(f"  Best CV NDCG : {best_cv_score:.4f}")
    print(f"  Best Params  : {best_params}")
    print(f"{'='*60}\n")

    # ── Save grid search results ──────────────────────────
    os.makedirs("../models", exist_ok=True)
    results_df.to_csv("../models/grid_search_results.csv", index=False)
    print("Grid search results saved → ../models/grid_search_results.csv")

    # ── Baseline model (original params) for comparison ──
    baseline_params = dict(
        n_estimators=100,
        learning_rate=0.01,
        num_leaves=15,
        min_child_samples=5,
    )
    baseline = train_final_model(baseline_params, train_df)
    baseline_ndcg = eval_model(baseline, test_df)
    print(f"\nBaseline test NDCG  : {baseline_ndcg:.4f}")

    # ── Tuned model ───────────────────────────────────────
    tuned = train_final_model(best_params, train_df)
    tuned_ndcg = eval_model(tuned, test_df)
    print(f"Tuned    test NDCG  : {tuned_ndcg:.4f}")
    print(f"Improvement         : {tuned_ndcg - baseline_ndcg:+.4f}")

    # ── Per-year breakdown ────────────────────────────────
    test_df_copy = test_df.copy()
    X_test = test_df_copy.drop(columns=COLUMNS_TO_DROP)
    test_df_copy["pred_score"] = tuned.predict(X_test)
    per_year = test_df_copy.groupby("Year").apply(calculate_ndcg, include_groups=False)
    print("\n--- Per-Year Test NDCG (tuned model) ---")
    print(per_year.to_string())

    # ── Persist tuned model ───────────────────────────────
    model_path = "../models/lgbm_ranker_tuned.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(tuned, f)
    print(f"\nTuned model saved → {model_path}")

    # ── Feature importance ────────────────────────────────
    fi = pd.Series(
        tuned.feature_importances_,
        index=test_df.drop(columns=COLUMNS_TO_DROP).columns,
    ).sort_values(ascending=False)
    print("\n--- Feature Importances (top 10) ---")
    print(fi.head(10).to_string())


if __name__ == "__main__":
    Main()