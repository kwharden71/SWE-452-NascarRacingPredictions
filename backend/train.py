# ==========================================================
# This file is designed to be ran locally to Train the model
# and create a pickle file.
# ==========================================================
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, label_ranking_average_precision_score
import numpy as np
import pickle
import os


def Main():
    # 1. Load Model and clean
    data = pd.read_csv("../data/nascar_driver_statistics.csv")    
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # 2. Calculate season rank by Points for each driver in the season
    data['Rank'] = data.groupby('Year')['Points'].rank(ascending=True, method='dense')
    data['Rank'] = (data['Rank'] - 1).astype(int)
    # 3. Sort by Year
    data = data.sort_values(by='Year').reset_index(drop=True)

    # 4. Group-Aware Split
    years = data['Year'].unique()
    train_years, test_years = train_test_split(years, test_size=0.2, random_state=42)

    train_df = data[data['Year'].isin(train_years)].copy()
    test_df = data[data['Year'].isin(test_years)].copy()

    # 5. Calculate Group Sizes
    q_train = train_df.groupby('Year').size().to_list()
    q_test = test_df.groupby('Year').size().to_list()

    # 6. Define Features and Target
    columns_to_drop = ["id", "Driver", "Points", "Year", "Rank"]
    
    X_train = train_df.drop(columns=columns_to_drop)
    y_train = train_df["Rank"]
    
    X_test = test_df.drop(columns=columns_to_drop)
    y_test = test_df["Rank"]

    # 7. Initialize and Train Ranker
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.01,
        num_leaves=15,
        min_child_samples=5,
        label_gain=[i for i in range(74)]
    )

    model.fit(
        X_train, y_train,
        group=q_train,
        eval_set=[(X_test, y_test)],
        eval_group=[q_test],
        eval_at=[1, 5, 10]
    )
    
    test_df['pred_score'] = model.predict(X_test)

    results = test_df.groupby('Year').apply(calculate_metrics, include_groups=False)
    print("\n--- Average Test Results ---")
    print(results.mean())

    models_dir = "../models"

    # Create folder 
    os.makedirs(models_dir, exist_ok=True)

    # Full path to model file
    model_path = os.path.join(models_dir, "lgbm_ranker.pkl")

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved successfully to {model_path}", flush=True)


def calculate_metrics(group):
    if len(group) < 2:
        return pd.Series({'ndcg': np.nan})

    # Invert rank: Higher value = Better finish (Rank 0 is the winner)
    # This makes the winner the most "relevant" (highest value)
    true_rel = (group['Rank'].max() - group['Rank']).values
    y_true = [true_rel] 
    y_score = [group['pred_score'].values]
    
    # NDCG is the primary metric for Ranking
    n_val = ndcg_score(y_true, y_score)
    
    return pd.Series({'ndcg': n_val})

if __name__ == "__main__":
    Main()