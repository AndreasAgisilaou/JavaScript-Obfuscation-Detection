"""FastAPI interface for the JavaScript obfuscation detection models.

Each of the three trainable models (RandomForest, XGBoost, neural network)
exposes one endpoint that trains on a named dataset folder under
detection/data/train/ and saves the resulting model under
detection/models/<algo>/ using the name you give it. Dataset and model name
are the only required inputs; everything else uses each model's built-in
defaults, so a model can be trained entirely from the /docs UI with no code
required.

Each of those endpoints also takes an optional `test_dataset`: if given, the
just-trained model is scored against detection/data/test/<test_dataset>/ and
an accuracy (0-1, the fraction of files correctly identified as obfuscated
or not) is returned alongside the saved model path. Leave it blank to skip
evaluation.

/openai/classify resolves whatever /sort couldn't: it classifies every file
in detection/data/sorted/<dataset>/unidentified/ with an OpenAI chat model
and moves each into obfuscated/ or non_obfuscated/ based on the verdict.
That's its only job, so it takes just `dataset`, `model`, and an
`X-OpenAI-Api-Key` header.

Run with (from the repo root, with reload support):
    uvicorn detection.main:app --reload

or directly (from inside detection/, no reload):
    python main.py
"""
import os
import shutil
from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

try:
    from detection import naural_network, open_ai, random_forest, xgboost_model
    from detection.features import (
        list_datasets, list_models, resolve_sorted_dirs, resolve_test_dirs, resolve_train_dirs,
        resolve_unsorted_dir,
    )
except ImportError:
    # Allows running this file directly (e.g. `python main.py` from inside
    # detection/), where "detection" isn't an importable package.
    import naural_network, open_ai, random_forest, xgboost_model
    from features import (
        list_datasets, list_models, resolve_sorted_dirs, resolve_test_dirs, resolve_train_dirs,
        resolve_unsorted_dir,
    )

# str Enums so /docs renders these as dropdowns instead of free text. Built
# from whatever folders/model files actually exist at startup (the train
# dataset dropdown falls back to "dataset_v1" if none are found yet; the
# others have no forced fallback since they're optional). Add a dataset or
# train a model to make it appear here, then restart the server.
DatasetName = Enum("DatasetName", {name: name for name in (list_datasets("train") or ["dataset_v1"])}, type=str)
TestDatasetName = Enum("TestDatasetName", {name: name for name in list_datasets("test")}, type=str)
UnsortedDatasetName = Enum("UnsortedDatasetName", {name: name for name in list_datasets("unsorted")}, type=str)
RandomForestModelName = Enum("RandomForestModelName", {name: name for name in list_models("random_forest")}, type=str)
XgboostModelName = Enum("XgboostModelName", {name: name for name in list_models("xgboost")}, type=str)
NeuralNetworkModelName = Enum("NeuralNetworkModelName", {name: name for name in list_models("neural_network")}, type=str)


def _require_simple_name(value: str, field_name: str) -> None:
    """dataset/model_name become path segments, so reject anything that
    could escape the intended data/models directory."""
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a plain name (no path separators): {value!r}")


def _require_dataset(dataset: str) -> None:
    _require_simple_name(dataset, "dataset")
    missing = [name for name, path in resolve_train_dirs(dataset).items() if not os.path.isdir(path)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{dataset}' is missing folder(s) under detection/data/train/{dataset}/: {', '.join(missing)}",
        )


def _require_test_dataset(test_dataset: str) -> None:
    missing = [name for name, path in resolve_test_dirs(test_dataset).items() if not os.path.isdir(path)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Test dataset '{test_dataset}' is missing folder(s) under detection/data/test/{test_dataset}/: {', '.join(missing)}",
        )


app = FastAPI(
    title="JavaScript Obfuscation Detection API",
    description="Train the RandomForest, XGBoost, and neural network detectors on a named dataset, or classify with an OpenAI model.",
)

TEST_DATASET_QUERY = Query(
    None,
    description="Optional: dataset folder under detection/data/test/ to evaluate the trained model against "
                "(accuracy = fraction of files correctly identified as obfuscated or not, 0-1). Leave blank to skip.",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


class ModelSavedResponse(BaseModel):
    model_path: str
    accuracy: Optional[float] = None


class NeuralNetworkSavedResponse(BaseModel):
    model_path: str
    scaler_path: str
    accuracy: Optional[float] = None


@app.post("/random-forest/train", response_model=ModelSavedResponse, tags=["training"])
def train_random_forest(
    dataset: DatasetName = Query(..., description="Dataset folder name under detection/data/train/"),
    model_name: str = Query(..., description="Name to save the trained model under in detection/models/random_forest/"),
    test_dataset: Optional[TestDatasetName] = TEST_DATASET_QUERY,
):
    dataset, test_dataset = dataset.value, test_dataset.value if test_dataset else None
    _require_dataset(dataset)
    _require_simple_name(model_name, "model_name")
    if test_dataset:
        _require_test_dataset(test_dataset)

    try:
        model_path = random_forest.train_model(dataset, model_name)
        accuracy = random_forest.evaluate_model(model_path, test_dataset) if test_dataset else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ModelSavedResponse(model_path=model_path, accuracy=accuracy)


@app.post("/xgboost/train", response_model=ModelSavedResponse, tags=["training"])
def train_xgboost(
    dataset: DatasetName = Query(..., description="Dataset folder name under detection/data/train/"),
    model_name: str = Query(..., description="Name to save the trained model under in detection/models/xgboost/"),
    test_dataset: Optional[TestDatasetName] = TEST_DATASET_QUERY,
):
    dataset, test_dataset = dataset.value, test_dataset.value if test_dataset else None
    _require_dataset(dataset)
    _require_simple_name(model_name, "model_name")
    if test_dataset:
        _require_test_dataset(test_dataset)

    try:
        model_path = xgboost_model.train_model(dataset, model_name)
        accuracy = xgboost_model.evaluate_model(model_path, test_dataset) if test_dataset else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ModelSavedResponse(model_path=model_path, accuracy=accuracy)


@app.post("/neural-network/train", response_model=NeuralNetworkSavedResponse, tags=["training"])
def train_neural_network(
    dataset: DatasetName = Query(..., description="Dataset folder name under detection/data/train/"),
    model_name: str = Query(..., description="Name to save the trained model under in detection/models/neural_network/"),
    test_dataset: Optional[TestDatasetName] = TEST_DATASET_QUERY,
):
    dataset, test_dataset = dataset.value, test_dataset.value if test_dataset else None
    _require_dataset(dataset)
    _require_simple_name(model_name, "model_name")
    if test_dataset:
        _require_test_dataset(test_dataset)

    try:
        model_path, scaler_path = naural_network.train_model(dataset, model_name)
        accuracy = naural_network.evaluate_model(model_path, scaler_path, test_dataset) if test_dataset else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return NeuralNetworkSavedResponse(model_path=model_path, scaler_path=scaler_path, accuracy=accuracy)


# --- Sorting (classify data/unsorted/ with 2-3 saved models and vote) ---

class SortResponse(BaseModel):
    obfuscated_dir: str
    non_obfuscated_dir: str
    unidentified_dir: str
    obfuscated_files: List[str]
    non_obfuscated_files: List[str]
    unidentified_files: List[str]


@app.post("/sort", response_model=SortResponse, tags=["sorting"])
def sort_dataset(
    dataset: UnsortedDatasetName = Query(..., description="Dataset folder name under detection/data/unsorted/"),
    random_forest_model: Optional[RandomForestModelName] = Query(
        None, description="RandomForest model under detection/models/random_forest/ to vote with. Set to opt this method in."
    ),
    xgboost_model_name: Optional[XgboostModelName] = Query(
        None, description="XGBoost model under detection/models/xgboost/ to vote with. Set to opt this method in."
    ),
    neural_network_model: Optional[NeuralNetworkModelName] = Query(
        None, description="Neural network model under detection/models/neural_network/ to vote with. Set to opt this method in."
    ),
):
    """Classifies every .js file in data/unsorted/<dataset>/ with 2 or 3
    selected models and copies each file into data/sorted/<dataset>/: into
    obfuscated/non_obfuscated only if every selected method agrees, otherwise
    into unidentified."""
    dataset = dataset.value
    selected = [
        (name, model_name.value) for name, model_name in [
            ("random_forest", random_forest_model),
            ("xgboost", xgboost_model_name),
            ("neural_network", neural_network_model),
        ] if model_name is not None
    ]
    if len(selected) < 2:
        raise HTTPException(
            status_code=400,
            detail="Select at least 2 of the 3 methods (random_forest_model, xgboost_model_name, neural_network_model) to sort by agreement.",
        )

    unsorted_dir = resolve_unsorted_dir(dataset)
    if not os.path.isdir(unsorted_dir):
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset}' is missing its folder under detection/data/unsorted/{dataset}/")

    try:
        predictions_by_method = {}
        for method, model_name in selected:
            if method == "random_forest":
                model_path = os.path.join(random_forest.MODELS_DIR, f"{model_name}.pkl")
                predictions_by_method[method] = random_forest.predict_directory(model_path, unsorted_dir)
            elif method == "xgboost":
                model_path = os.path.join(xgboost_model.MODELS_DIR, f"{model_name}.pkl")
                predictions_by_method[method] = xgboost_model.predict_directory(model_path, unsorted_dir)
            else:
                model_path = os.path.join(naural_network.MODELS_DIR, f"{model_name}.h5")
                scaler_path = os.path.join(naural_network.MODELS_DIR, f"{model_name}_scaler.pkl")
                predictions_by_method[method] = naural_network.predict_directory(model_path, scaler_path, unsorted_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    all_filenames = sorted(set().union(*(set(p) for p in predictions_by_method.values())))

    sorted_dirs = resolve_sorted_dirs(dataset)
    for path in sorted_dirs.values():
        os.makedirs(path, exist_ok=True)

    obfuscated_files, non_obfuscated_files, unidentified_files = [], [], []
    for filename in all_filenames:
        votes = {predictions_by_method[method].get(filename) for method, _ in selected}
        if votes == {1}:
            category, bucket = "obfuscated", obfuscated_files
        elif votes == {0}:
            category, bucket = "non_obfuscated", non_obfuscated_files
        else:
            category, bucket = "unidentified", unidentified_files

        shutil.copy(os.path.join(unsorted_dir, filename), os.path.join(sorted_dirs[category], filename))
        bucket.append(filename)

    return SortResponse(
        obfuscated_dir=sorted_dirs["obfuscated"],
        non_obfuscated_dir=sorted_dirs["non_obfuscated"],
        unidentified_dir=sorted_dirs["unidentified"],
        obfuscated_files=obfuscated_files,
        non_obfuscated_files=non_obfuscated_files,
        unidentified_files=unidentified_files,
    )


# --- OpenAI resolution (classifies whatever /sort left in unidentified/) ---

class OpenAIResolveResponse(BaseModel):
    obfuscated_dir: str
    non_obfuscated_dir: str
    obfuscated_files: List[str]
    non_obfuscated_files: List[str]
    unresolved_files: List[str]


@app.post("/openai/classify", response_model=OpenAIResolveResponse, tags=["sorting"])
def classify_with_openai(
    dataset: UnsortedDatasetName = Query(
        ..., description="Dataset folder name under detection/data/unsorted/ (its sorted/<dataset>/unidentified/ files will be resolved)"
    ),
    api_key: str = Header(
        ..., alias="X-OpenAI-Api-Key", description="OpenAI API key, used only for this request and not persisted"
    ),
    model: str = Query("gpt-4o-mini", description="OpenAI chat model to use"),
):
    dataset = dataset.value
    sorted_dirs = resolve_sorted_dirs(dataset)
    if not os.path.isdir(sorted_dirs["unidentified"]):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{dataset}' has no unidentified files yet under detection/data/sorted/{dataset}/unidentified/ - run /sort first.",
        )

    try:
        obfuscated_files, non_obfuscated_files, unresolved_files = open_ai.resolve_unidentified(dataset, api_key, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return OpenAIResolveResponse(
        obfuscated_dir=sorted_dirs["obfuscated"],
        non_obfuscated_dir=sorted_dirs["non_obfuscated"],
        obfuscated_files=obfuscated_files,
        non_obfuscated_files=non_obfuscated_files,
        unresolved_files=unresolved_files,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
