# JavaScript-Obfuscation-Detection

Classifies JavaScript files as **Obfuscated** or **NonObfuscated** using the same set of hand-crafted lexical/statistical heuristics (entropy, line length, keyword frequency, compression ratio, etc.) fed into four different approaches, plus a FastAPI interface to train any of the trainable ones over HTTP.

## Approaches

The RandomForest and XGBoost approaches share the same feature extraction code in [detection/features.py](detection/features.py). The neural network uses its own separate feature set (defined in the script itself).

| Script | Approach |
| --- | --- |
| [detection/random_forest.py](detection/random_forest.py) | `RandomForestClassifier` (scikit-learn) |
| [detection/xgboost_model.py](detection/xgboost_model.py) | `XGBClassifier` (XGBoost) |
| [detection/naural_network.py](detection/naural_network.py) | Small feed-forward neural network (Keras/TensorFlow) |
| [detection/open_ai.py](detection/open_ai.py) | Zero-shot classification via an OpenAI chat model (`gpt-4o-mini` by default) — resolves whatever the other three couldn't agree on, no training required |

## Datasets

The three trainable scripts read from a named dataset folder under `detection/data/train/`:

```
detection/data/train/<dataset>/
├── obfuscated/       # .js files
└── non_obfuscated/   # .js files
```

`detection/data/test/<dataset>/` follows the same `obfuscated`/`non_obfuscated` shape and is used for optional evaluation (see below).

Training uses each model's built-in defaults (no exposed hyperparameters) and always fits + saves the model; scoring it against a test dataset is optional and only happens if you ask for it.

`detection/data/unsorted/<dataset>/` is a flat folder of `.js` files not yet labeled. The pipeline for sorting them is two steps:

1. `/sort` classifies them with 2-3 saved models and copies each into `detection/data/sorted/<dataset>/{obfuscated,non_obfuscated,unidentified}/` — `unidentified` is whatever those models disagreed on.
2. `/openai/classify` resolves `unidentified/` with an OpenAI model, moving each file into `obfuscated/` or `non_obfuscated/` once it has a verdict.

## Setup

```
pip install -r detection/requirements.txt
```

`open_ai.py` (and the `/openai/classify` API endpoint) needs an OpenAI API key. It targets the legacy `openai<1.0` SDK, which `detection/requirements.txt` pins to `0.28.1` (the last release before the 1.0 rewrite). The key is passed per-request (`api_key`), never stored.

## Running a script directly

Each trainable script has `DEFAULT_DATASET` / `DEFAULT_MODEL_NAME` constants near the top (both default to `"dataset_v1"` / `"<algo>_model"`) — populate `detection/data/train/dataset_v1/{obfuscated,non_obfuscated}` with `.js` files, then:

```
python -m detection.random_forest
python -m detection.xgboost_model
python -m detection.naural_network
python -m detection.open_ai   # requires OPENAI_API_KEY; resolves data/sorted/dataset_v1/unidentified/
```

## Running the API

```
uvicorn detection.main:app --reload
```

(or `python main.py` from inside `detection/`, no reload). Visiting `http://127.0.0.1:8000/` redirects to the interactive docs at `/docs`, which lets you fill in and try every endpoint directly from the browser — no code required.

### Endpoints

All parameters are query parameters rather than a JSON body (the routes are still POST, since they train models and write files to disk).

**Training** (tag `training`) — each trains on `dataset` (folder under `detection/data/train/`, checked to actually have `obfuscated`/`non_obfuscated` subfolders) and saves the model under `detection/models/<algo>/<model_name>`, using the model's built-in defaults for everything else. `dataset` and `model_name` are both validated as plain names (no path separators).

- `POST /random-forest/train` — saves to `detection/models/random_forest/<model_name>.pkl`.
- `POST /xgboost/train` — saves to `detection/models/xgboost/<model_name>.pkl`.
- `POST /neural-network/train` — saves to `detection/models/neural_network/<model_name>.h5` (+ a matching `<model_name>_scaler.pkl`).

Each also takes an optional `test_dataset` (folder under `detection/data/test/`, same dropdown mechanism as `dataset`). If given, the just-trained model is immediately scored against it and the response includes `accuracy` — the fraction of files correctly identified as obfuscated or not, from 0 to 1. Leave it blank to skip evaluation, in which case `accuracy` comes back `null`.

**Sorting** (tag `sorting`) — the two-step pipeline described above.

- `POST /sort` — takes `dataset` (folder under `detection/data/unsorted/`) plus three optional model-name dropdowns — `random_forest_model`, `xgboost_model_name`, `neural_network_model` — each populated from that algorithm's saved models; setting one opts that method into the vote. At least 2 of the 3 must be set. Classifies every `.js` file in `detection/data/unsorted/<dataset>/` and copies each into `detection/data/sorted/<dataset>/obfuscated|non_obfuscated/` only if **every** selected method agrees, otherwise into `unidentified/`. Files are copied, not moved — the originals stay in `unsorted`.
- `POST /openai/classify` — takes `dataset` (same dropdown as `/sort`), `api_key` (used only for that request, never persisted — note it's part of the URL since it's a query parameter, so avoid this endpoint if the API is exposed beyond local use), and `model`. Classifies every file in `detection/data/sorted/<dataset>/unidentified/` and **moves** it into `obfuscated/` or `non_obfuscated/`; a file is left in `unidentified/` only if every chunk of it fails to get a response from OpenAI.

**Other**

- `GET /health` — liveness check.

These endpoints run synchronously and block until they finish — training, sorting, or a full OpenAI resolution pass can take a while. For large datasets or high request volume, consider moving the calls onto a background task queue instead of calling them directly from the request handler.

## Outputs

Trained models (and, for the neural network, its scaler) are written to `detection/models/<algo>/`. Sorted files are copied to `detection/data/sorted/<dataset>/`. None of these — nor the dataset files under `detection/data/` — are committed to the repo.
