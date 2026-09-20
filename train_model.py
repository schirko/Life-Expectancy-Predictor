"""
Train the Random Forest life expectancy model and save it to disk for the
Streamlit app to load.

Run with:  python train_model.py
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_prep import PREDICTOR_COLS, TARGET_COL, feature_ranges, get_model_ready_df

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Life_Expectancy_WHO_Data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "rf_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "model", "metadata.json")


def main():
    print("Loading and cleaning data...")
    df = get_model_ready_df(DATA_PATH)
    df = df.dropna(subset=PREDICTOR_COLS + [TARGET_COL])
    print(f"Model-ready rows: {len(df)}")

    x = df[PREDICTOR_COLS].values
    y = df[TARGET_COL].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test R2:   {r2:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    importances = dict(
        sorted(
            zip(PREDICTOR_COLS, model.feature_importances_.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "predictor_cols": PREDICTOR_COLS,
        "target_col": TARGET_COL,
        "test_r2": r2,
        "test_mae": mae,
        "test_rmse": rmse,
        "feature_importances": importances,
        "feature_ranges": feature_ranges(df),
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")

    # Also save the cleaned/winsorized dataframe for the dashboard page,
    # so the app doesn't need to re-run cleaning on every load.
    clean_path = os.path.join(os.path.dirname(__file__), "data", "clean_data.csv")
    df.to_csv(clean_path, index=False)
    print(f"Saved cleaned dataset to {clean_path}")


if __name__ == "__main__":
    main()
