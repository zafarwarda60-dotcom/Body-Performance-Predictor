# Body Performance Predictor

## Project description

This project is a machine-learning web application that uses body measurements to estimate body-fat percentage and predict an exercise performance class from A to D. It includes a scikit-learn training pipeline, saved prediction models, a FastAPI backend, and a simple browser interface. These outputs are estimates for information only and are **not health or medical advice**.

## Dataset

The project uses the [Body Performance dataset on Kaggle](https://www.kaggle.com/datasets/kukuroo3/body-performance-data), supplied locally as `bodyPerformance.csv`. The training script removes incomplete records, duplicate rows, and rows where diastolic pressure is zero or systolic pressure is not greater than diastolic pressure. On the included file, this leaves **13,387 records**: one duplicate and five invalid blood-pressure rows are removed.

## Setup and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py
uvicorn main:app --reload
```

### Run in Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python train_models.py
python -m uvicorn main:app --reload
```

### Run from VS Code

1. Open the project folder in VS Code.
2. Open **Terminal > New Terminal**.
3. Run the setup commands above for PowerShell or Command Prompt.
4. Open <http://127.0.0.1:8000> in a browser.

### Run the API without the browser interface

After starting the server, use the interactive API documentation at <http://127.0.0.1:8000/docs> to send prediction requests. You can also check <http://127.0.0.1:8000/health> to confirm that the API is running.

Open <http://127.0.0.1:8000> for the web page, <http://127.0.0.1:8000/docs> for interactive API docs, <http://127.0.0.1:8000/health> to check server status, or <http://127.0.0.1:8000/metrics> for the chart data. The models are saved as joblib files under `models/`.

## Evaluation

The current saved models use an 80/20 holdout with `random_state=42`. On the included dataset and the current training script, the results are:

| Task | Holdout metric | Five-fold cross-validation |
| --- | ---: | ---: |
| Body-fat regression | R² **0.7425**, MAE **2.84 percentage points** | Mean R² **0.7255** (std 0.0100) |
| Performance class | Accuracy **75.21%** | Mean accuracy **73.65%** (std 0.97%) |

The classification confusion matrix, feature lists, cleaning counts, and body-fat feature importances are saved in `models/metrics.json` and printed by `python train_models.py`. The strongest body-fat regression feature in this run is broad-jump distance. Metrics can vary with scikit-learn version or training changes; `metrics.json` records the results for the saved models.

## API examples

Both prediction endpoints accept the following measurements. Inputs are limited to ranges observed in the cleaned training data. The systolic value must be greater than diastolic.

```json
{
  "age": 28,
  "gender": "M",
  "height_cm": 173,
  "weight_kg": 70,
  "diastolic": 75,
  "systolic": 120,
  "grip_force": 40,
  "sit_bend_forward": 15,
  "sit_ups_count": 45,
  "broad_jump_cm": 210
}
```

- `POST /predict/regression` returns a body-fat estimate.
- `POST /predict/classification` returns a performance class. It accepts optional `body_fat` (percentage); if omitted, the regression model supplies an estimate.
- `GET /health` returns `{"status":"ok"}`.

## Project structure

```text
main.py                 FastAPI routes and input validation
train_models.py         Cleaning, training, evaluation, and model export
bodyPerformance.csv     Training data
models/                 Saved joblib pipelines and metrics.json
static/                 Single-page app, styles, and browser code
tests/test_api.py       API validation and prediction tests
requirements.txt        Exact installed package versions
```

## Tests

```powershell
python -m pytest
```

The tests cover valid regression/classification requests, supplied or estimated body fat, invalid values, and the health route.
