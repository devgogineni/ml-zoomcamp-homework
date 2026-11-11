import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

DATA_PATH = "data/solar_energy_production_data_for_2025.csv"
TARGET = "solar_energy_generated"
SEED = 1
MODEL_PATH = "model/linear_regression.bin"
WEIGHTS = [0.01,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0]

# training parameters per model family:
#   estimator   model class to instantiate
#   fixed       parameters passed to every candidate
#   param_grid  parameters to tune; every combination is tried (grid search)
#   n_iter      if set, only n_iter combinations are randomly sampled (random search)
MODELS = {
    "Ridge": {
        "estimator": Ridge,
        "fixed": {},
        "param_grid": {"alpha": WEIGHTS},
    },
    "DecisionTree": {
        "estimator": DecisionTreeRegressor,
        "fixed": {"random_state": SEED},
        "param_grid": {
            "max_depth": [2, 3, 4, 5, 6, 8, 10],
            "min_samples_leaf": [1, 5, 10],
        },
    },
    "RandomForest": {
        "estimator": RandomForestRegressor,
        "fixed": {"random_state": SEED, "n_jobs": -1},
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 3, 5],
        },
    },
    # the full xgboost grid is too large, so n_iter configs are sampled from it
    "XGBoost": {
        "estimator": XGBRegressor,
        "fixed": {"random_state": SEED, "n_jobs": -1},
        "param_grid": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4, 6],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 5, 10],
        },
        "n_iter": 60,
    },
}

NUMERICAL = [
    "tempmax", "tempmin", "temp", "dew", "humidity", "precip", "precipprob",
    "precipcover", "cloudcover",
]
# same-day solarradiation, solarenergy and uvindex measure the sunlight itself,
# so they are not available in advance and are excluded to avoid target leakage
CONDITION_FLAGS = ["clear", "partially_cloudy", "overcast", "rain", "snow"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()

    # cyclical encoding of the day of year so Dec 31 sits next to Jan 1
    day_of_year = df["date"].dt.day_of_year
    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    # "conditions" is a comma separated list, e.g. "Snow, Rain, Overcast"
    conditions = df["conditions"].str.lower().str.replace(" ", "_")
    for flag in CONDITION_FLAGS:
        df[flag] = conditions.str.contains(flag).astype(int)

    return df


def prepare_data(df):
    df = engineer_features(df)
    features = NUMERICAL + ["day_sin", "day_cos"] + CONDITION_FLAGS

    # 60% train / 20% validation / 20% test
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=SEED)

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient="records"))
    X_val = dv.transform(df_val[features].to_dict(orient="records"))
    X_test = dv.transform(df_test[features].to_dict(orient="records"))

    # fit the scaler on train only to avoid leaking validation/test statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, dv, scaler


def candidates(spec):
    if "n_iter" in spec:
        return ParameterSampler(spec["param_grid"], n_iter=spec["n_iter"], random_state=SEED)
    return ParameterGrid(spec["param_grid"])


def train(spec, X_train, y_train, params):
    model = spec["estimator"](**spec["fixed"], **params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, high_threshold):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # AUC needs a binary target: a "high production" day is one above the
    # threshold, and the predicted energy is used as the ranking score
    auc = roc_auc_score(y > high_threshold, y_pred)
    return rmse, r2_score(y, y_pred), auc


def save_model(model, dv, scaler, path=MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        pickle.dump((dv, scaler, model), f_out)


if __name__ == "__main__":
    df = load_data()
    print(f"loaded {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    X_train, X_val, X_test, y_train, y_val, y_test, dv, scaler = prepare_data(df)
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # threshold taken from train only so val/test labels don't leak into it
    high_threshold = np.median(y_train)
    print(f"high production day: > {high_threshold:.2f}")

    splits = [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]

    def run(label, model):
        metrics = {name: evaluate(model, X, y, high_threshold) for name, X, y in splits}
        print(label)
        for name, (rmse, r2, auc) in metrics.items():
            print(f"  {name:5s} rmse={rmse:.3f} r2={r2:.3f} auc={auc:.3f}")
        return metrics

    # (name, model, metrics) per model family, each tuned on validation rmse
    best_per_family = []

    for family, spec in MODELS.items():
        print(f"== {family} ==")
        results = []
        for params in candidates(spec):
            label = " ".join(f"{k}={v}" for k, v in params.items())
            model = train(spec, X_train, y_train, params)
            results.append((label, model, run(label, model)))
        best_per_family.append((family, *min(results, key=lambda r: r[2]["val"][0])))

    # select on validation RMSE so the test set stays an unbiased final check
    print("== best per model (selected on validation rmse) ==")
    for family, params, _, metrics in best_per_family:
        print(f"{family:12s} {params}")
        print("  val  rmse={:.3f} r2={:.3f} auc={:.3f}".format(*metrics["val"]))
        print("  test rmse={:.3f} r2={:.3f} auc={:.3f}".format(*metrics["test"]))

    family, params, model, metrics = min(best_per_family, key=lambda r: r[3]["val"][0])
    print(f"chosen model: {family} ({params})")

    feature_names = dv.get_feature_names_out()
    if hasattr(model, "coef_"):
        scores = model.coef_
        print("top coefficients (standardized features):")
    else:
        scores = model.feature_importances_
        print("top feature importances:")
    for feature, score in sorted(zip(feature_names, scores), key=lambda c: -abs(c[1]))[:8]:
        print(f"  {feature:18s} {score:+.3f}")

    save_model(model, dv, scaler)
    print(f"model saved to {MODEL_PATH}")
