# Solar energy generation prediction

Predicts the daily solar energy generated (`solar_energy_generated`) from weather and calendar features, using the 2025 daily dataset in `data/solar_energy_production_data_for_2025.csv` (365 rows, one per day, no missing values).

Four regression model families are trained and tuned, compared on a held-out validation set, and the best one is saved to `model/linear_regression.bin`. A FastAPI app serves the saved model through a web form and a JSON API, and is set up for deployment on [Fly.io](https://fly.io).

## Files

| file | purpose |
|---|---|
| `training.py` | full pipeline: load, prepare, tune, compare, save |
| `training.ipynb` | the same code split into cells (copy of `training.py`, kept in sync manually) |
| `model/linear_regression.bin` | pickled `(DictVectorizer, StandardScaler, model)` tuple for the chosen model |
| `pyproject.toml`, `uv.lock` | project dependencies and the exact locked versions, managed with [uv](https://docs.astral.sh/uv/) |
| `app/main.py` | FastAPI backend: loads the saved model and exposes the prediction API |
| `app/static/index.html` | web form for entering the model inputs |
| `Dockerfile` | container image that runs the training script |
| `Dockerfile.serve` | container image that serves the web app and API |
| `fly.toml` | Fly.io deployment config (uses `Dockerfile.serve`) |
| `.dockerignore` | files kept out of both Docker builds |

## Setup

Dependencies are managed with uv and declared in `pyproject.toml` (Python 3.12 or newer):
- **Runtime:** `numpy`, `pandas`, `scikit-learn`, `xgboost` for training, plus `fastapi` and `uvicorn` for the web app.
- **Dev group:** `jupyter`, `ipykernel`, for running the notebook.

From the `midterm-project/` directory, create the virtual environment (`.venv`) with:

```bash
uv sync
```

## Running

Run everything from the `midterm-project/` directory, because the data and model paths are relative.

**Script**
```bash
uv run python training.py
```

**Notebook:** open `training.ipynb` with the `midterm-project/.venv` interpreter and run the cells from top to bottom.

**Docker**

```bash
docker build -t solar-training .
docker run --rm -v "$PWD/model:/app/model" solar-training
```

This uses `Dockerfile`, the training image. It is based on `python:3.12-slim` and installs the locked dependencies with `uv sync --frozen --no-dev`, so the Jupyter packages are left out. It also installs `libgomp1`, which XGBoost needs on Linux. The dependency layer is cached until `pyproject.toml` or `uv.lock` changes. The default command is `python training.py`.

Mount `model/` as shown above to get the trained `linear_regression.bin` back on your machine. Without the mount, the model stays inside the container and is lost when it is removed. The training container only trains. To serve predictions, see the next section.

## Web app and API

The FastAPI app in `app/main.py` loads `model/linear_regression.bin` at startup, so train the model first. It serves both the API and the web form from one process.

```bash
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000 for the form, or http://localhost:8000/docs for the interactive API docs.

### Form inputs
The form asks for every input the model uses:
- **Date:** defaults to today. The model uses only the day of the year (see below).
- **Nine numeric fields:** `tempmax`, `tempmin`, `temp`, `dew`, `humidity`, `precip`, `precipprob`, `precipcover`, `cloudcover`. They start at the training medians.
- **Sky:** a dropdown with Clear, Partially cloudy or Overcast.
- **Precipitation:** checkboxes for Rain and Snow.

Conditions are split this way because every day in the training data has exactly one sky condition plus optional rain and/or snow, so the form cannot produce a combination the model never saw (such as Clear with Rain).

**Why the date is an input.** The model never sees the raw date. Training turns it into two seasonal features, `day_sin` and `day_cos`, because the sun's height and the day length change through the year and drive solar output. In 5-fold cross-validation, removing them raised the Ridge RMSE from 3.59 to 4.38 (R² from 0.883 to 0.826), and `day_cos` correlates -0.77 with the target. So the model needs the time of year to predict, and the date field lets you ask what a given set of weather values would give in a different season.

The form is built from `GET /api/features`, so the fields always match what the backend accepts.

### Endpoints

| endpoint | purpose |
|---|---|
| `GET /` | the web form |
| `GET /api/features` | describes the form fields (names, labels, bounds, defaults, condition options) |
| `POST /api/predict` | returns `{"solar_energy_generated": <number>}` |
| `GET /health` | health check, used by Fly.io |

Example:

```bash
curl -X POST localhost:8000/api/predict -H 'content-type: application/json' -d '{
  "date": "2025-06-15", "tempmax": 22, "tempmin": 12, "temp": 17, "dew": 9,
  "humidity": 60, "precip": 0, "precipprob": 0, "precipcover": 0, "cloudcover": 20,
  "sky": "clear", "precipitation": []
}'
```

### Behaviour
- **Validation:** invalid input returns HTTP 422. Percentages must be 0 to 100, no field may be missing, `sky` must be one of the three options, and the temperatures must satisfy `tempmin <= temp <= tempmax`.
- **Same features as training:** the backend calls `engineer_features` from `training.py`, so the day-of-year and condition encoding is identical at serving time.
- **Non-negative output:** the linear model can predict below zero, so negative predictions are clamped to 0.
- **Units:** the form labels temperatures as °C, precipitation as mm and the rest as %. These are assumptions from the value ranges, because the dataset does not state units. The predicted solar energy is shown in kW in the web form; the API returns the bare number.

### Serving with Docker

```bash
docker build -f Dockerfile.serve -t solar-serving .
docker run --rm -p 8080:8080 solar-serving
```

Open http://localhost:8080. `Dockerfile.serve` copies in `training.py`, `app/` and the trained `model/linear_regression.bin`, installs the locked dependencies with uv, and runs `uvicorn` on port 8080 as a non-root user. The model is baked into the image, so retrain and rebuild to ship a new model. On a local test the container used about 145 MiB of memory.

## Deploying to Fly.io

The app is deployed at https://solar-energy-predictor.fly.dev/ (app `solar-energy-predictor`, region `iad`).

`fly.toml` points at `Dockerfile.serve` and exposes port 8080 over HTTPS, with `/health` as the health check. Machines stop when idle and start on the next request, so the first request after a quiet period takes a few seconds. The VM is `shared-cpu-1x` with 512 MB. Fly started two machines for high availability; run `fly scale count 1` if you want one.

`fly.toml` also sets a per-machine concurrency limit (soft 25, hard 50 simultaneous requests). Above the soft limit Fly prefers other machines, and above the hard limit it queues or rejects requests. This limits simultaneous load, not requests per client, so it is not per-client rate limiting.

### First deploy
1. Install flyctl and log in: `fly auth login`.
2. Change `app` in `fly.toml` to a name that is unique on Fly.io, and set `primary_region` to one near you.
3. Make sure `model/linear_regression.bin` is the model you want to ship (run `training.py`).
4. Deploy from `midterm-project/`:
   ```bash
   fly launch --no-deploy --copy-config
   fly deploy
   ```

### Updating
Run `fly deploy` again from `midterm-project/` after changing the code, `fly.toml` or the model. The model is baked into the image, so a new model only goes live after a redeploy.

### Useful commands
- `fly status`: machine state and health checks.
- `fly logs`: application logs.
- `fly apps destroy solar-energy-predictor`: remove the app.

## Training process

### 1. Data loading
The CSV is read with the `date` column parsed, duplicate dates dropped and rows sorted by date.

### 2. Features
- **Weather (numeric):** `tempmax`, `tempmin`, `temp`, `dew`, `humidity`, `precip`, `precipprob`, `precipcover`, `cloudcover`.
- **Calendar:** day of year encoded cyclically as `day_sin` and `day_cos`, so Dec 31 sits next to Jan 1.
- **Conditions:** the comma-separated `conditions` text (for example `"Snow, Rain, Overcast"`) becomes five 0/1 flags: `clear`, `partially_cloudy`, `overcast`, `rain`, `snow`.

**Excluded to avoid target leakage:** same-day `solarradiation`, `solarenergy` and `uvindex`. They measure the sunlight the panels convert (correlation with the target is about 0.93 to 0.97), so they would not be available when forecasting. With them the model scored R² of about 0.95, so the high score was not a sign of overfitting. Excluding them gives a more realistic estimate of forecasting skill (R² of about 0.87 to 0.89).

### 3. Split and scaling
- Random 60% train / 20% validation / 20% test split (219 / 73 / 73 rows), `random_state=1`.
- A `DictVectorizer` builds the feature matrix and a `StandardScaler` standardizes it. Both are **fit on the training set only**, so validation and test statistics do not leak into preprocessing.

### 4. Models and tuning
Each family is tuned by training every candidate configuration on the training set and scoring it on the validation set.

The training parameters for every family live in one `MODELS` dict in `training.py`. Each entry has:
- `estimator`: the model class.
- `fixed`: parameters used for every candidate, such as `random_state` and `n_jobs`.
- `param_grid`: the parameters to tune.
- `n_iter` (optional): if set, that many combinations are sampled at random (random search) instead of trying all of them (grid search). Only XGBoost sets it.

One shared loop tunes every entry, so adding a model family means adding one entry to `MODELS`.

| model | search | candidates |
|---|---|---|
| Ridge | grid over `alpha` (`WEIGHTS`) | 14 values: 0.01, 0.09, 0.1, 0.2 to 1.0 in steps of 0.1, 2.0, 3.0 |
| DecisionTreeRegressor | grid over `max_depth` x `min_samples_leaf` | 7 x 3 = 21 |
| RandomForestRegressor | grid over `n_estimators` x `max_depth` x `min_samples_leaf` | 2 x 3 x 3 = 18 |
| XGBRegressor | random search (`ParameterSampler`) over `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` | 60 sampled from about 15,000 |

The grids are defined in `MODELS`, and together with the `SEED` constant at the top of `training.py` they make runs reproducible.

## Model selection approach

### Metrics
For every configuration, on train, validation and test:
- **RMSE** (lower is better): the primary metric and the selection criterion.
- **R²**: share of variance explained.
- **AUC**: needs a binary target, so a day counts as "high production" when its actual output is above the **median of the training target** (14.93), and the model's predicted energy is the ranking score. The threshold comes from the training set only. AUC measures ranking, not accuracy, and saturates easily on a median split, so it is a secondary metric.

### Selection rule
1. Within each family, the configuration with the **lowest validation RMSE** is that family's winner.
2. Across families, the winner with the **lowest validation RMSE** is the chosen model.
3. The test set is never used for selection. It is only reported, as a final check on the chosen model.
4. The chosen model is saved together with its vectorizer and scaler.

### Results (last run)

| model | best parameters | val RMSE | val R² | test RMSE | test R² | test AUC |
|---|---|---|---|---|---|---|
| Ridge | alpha=0.9 | 3.637 | 0.865 | 3.849 | 0.878 | 0.973 |
| DecisionTree | max_depth=5, min_samples_leaf=1 | 5.512 | 0.690 | 4.807 | 0.810 | 0.969 |
| RandomForest | n_estimators=200, max_depth=None, min_samples_leaf=1 | 4.466 | 0.797 | 3.722 | 0.886 | 0.972 |
| XGBoost | n_estimators=200, learning_rate=0.1, max_depth=3, min_child_weight=5, subsample=1.0, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=5 | 3.806 | 0.852 | 3.593 | 0.894 | 0.984 |

**Chosen model: Ridge (alpha=0.9)**, the lowest validation RMSE.

## Limitations

- **Small validation set.** With 73 rows per split, validation and test rankings disagree: Ridge wins on validation, XGBoost is ahead on test. The Ridge versus XGBoost choice is not clear-cut.
- **Selection bias.** Picking the best of many configurations on one small validation set favours lucky configurations, especially for XGBoost (60 trials).
- **Random split.** The split is random, not chronological, so neighbouring days can fall in different sets. It does not test performance across seasons.
- **One year of data.** The model has seen each season once.
- **Unstable Ridge coefficients.** `temp`, `tempmax`, `tempmin` and `dew` are strongly collinear, so individual coefficients are not interpretable even though predictions are fine.
- **Saved filename.** The model path is `linear_regression.bin` regardless of which family wins.
- **Model baked into the serving image.** Retraining does not change a deployed app until you rebuild and redeploy.
- **No authentication or per-client rate limiting.** The API is open to anyone who can reach it. The Fly concurrency limit only caps simultaneous requests per machine.

## Possible improvements

- Replace the single split with k-fold cross-validation over all 365 rows for a steadier ranking.
- Add a chronological split (for example train on Jan to Sep, test on Oct to Dec).
- Widen the Ridge `alpha` range if needed. The best value (0.9) is inside the current grid, but validation RMSE changes very little across alphas, so Ridge is insensitive to this setting.
- Prune collinear temperature features.
