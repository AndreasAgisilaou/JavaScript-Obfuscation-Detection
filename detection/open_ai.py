"""Resolves files the local models disagreed on — the .js files sitting in
detection/data/sorted/<dataset>/unidentified/ — by classifying each with an
OpenAI chat model and moving it into obfuscated/ or non_obfuscated/ based on
the verdict. This is the only thing this module does; it takes no dataset
directories of its own, only an API key and a model name.

Note: this targets the legacy `openai<1.0` SDK (`openai.ChatCompletion.create`
/ `openai.error.InvalidRequestError`). Install that version, or port the
calls to the current `openai` client, before running.
"""
import os
import shutil
import time
import traceback

import openai

try:
    from detection.features import resolve_sorted_dirs
except ImportError:
    # Allows running this file directly (e.g. `python open_ai.py` from
    # inside detection/), where "detection" isn't an importable package.
    from features import resolve_sorted_dirs

# --- Configuration ---
DEFAULT_DATASET = "dataset_v1"

MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 16000  # Safe limit for the model's input
CHUNK_SIZE = 3000  # Characters per chunk sent to the model
RETRY_LIMIT = 3
RETRY_WAIT_SECONDS = 60


def get_llm_prediction(script_text, api_key, model=MODEL_NAME, max_tokens=MAX_TOKENS,
                        retry_limit=RETRY_LIMIT, retry_wait_seconds=RETRY_WAIT_SECONDS):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI that categorizes JavaScript code as Obfuscated or "
                "NonObfuscated. Minified scripts should also be considered Obfuscated. "
                "JSON-like files (structured, readable data) should be categorized as NonObfuscated."
            ),
        },
        {
            "role": "user",
            "content": f"""
    Analyze the following JavaScript code and determine if it is Obfuscated or NonObfuscated:
    - Consider scripts with:
      - Very long lines, unusual characters, or escape sequences as Obfuscated.
      - JSON-like or clearly formatted scripts as NonObfuscated.
    Please Triple check your results

    JavaScript code:
    {script_text}

    Your response should only be one word: 'Obfuscated' or 'NonObfuscated'.
    """,
        },
    ]

    for attempt in range(retry_limit):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=5,
                temperature=0.0,
                api_key=api_key,
            )
            return response.choices[0].message['content'].strip()
        except openai.error.InvalidRequestError as e:
            print(f"Error: {e}. Input too large; truncating input.")
            return get_llm_prediction(script_text[:max_tokens], api_key, model=model, max_tokens=max_tokens,
                                       retry_limit=retry_limit, retry_wait_seconds=retry_wait_seconds)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry_limit} failed: {e}")
            if attempt < retry_limit - 1:
                print(f"Retrying in {retry_wait_seconds} seconds...")
                time.sleep(retry_wait_seconds)
            else:
                print("Maximum retries reached. Skipping this file.")
                return "Error"


def classify_file(file_path, api_key, model=MODEL_NAME, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS):
    """Reads a file and returns the majority-vote LLM verdict across chunks
    ('Obfuscated' or 'NonObfuscated'), or None if every chunk failed (the
    caller should leave the file unresolved rather than guess)."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read().strip()

    predictions = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        prediction = get_llm_prediction(chunk, api_key, model=model, max_tokens=max_tokens)
        if prediction != "Error":
            predictions.append(prediction)

    if not predictions:
        return None
    return "Obfuscated" if predictions.count("Obfuscated") > predictions.count("NonObfuscated") else "NonObfuscated"


def resolve_unidentified(dataset, api_key, model=MODEL_NAME):
    """Classifies every file in data/sorted/<dataset>/unidentified/ and moves
    it into obfuscated/ or non_obfuscated/ based on the verdict. Files where
    every chunk failed are left in unidentified/. Returns
    (obfuscated_files, non_obfuscated_files, unresolved_files)."""
    sorted_dirs = resolve_sorted_dirs(dataset)
    unidentified_dir = sorted_dirs["unidentified"]

    for path in (sorted_dirs["obfuscated"], sorted_dirs["non_obfuscated"], unidentified_dir):
        os.makedirs(path, exist_ok=True)

    obfuscated_files, non_obfuscated_files, unresolved_files = [], [], []
    for filename in sorted(os.listdir(unidentified_dir)):
        if not filename.endswith('.js'):
            continue
        file_path = os.path.join(unidentified_dir, filename)

        try:
            print(f"Classifying {file_path}...")
            verdict = classify_file(file_path, api_key, model=model)
        except Exception:
            print(f"Error classifying {file_path}: {traceback.format_exc()}")
            verdict = None

        if verdict is None:
            unresolved_files.append(filename)
            continue

        category, bucket = (
            ("obfuscated", obfuscated_files) if verdict == "Obfuscated" else ("non_obfuscated", non_obfuscated_files)
        )
        shutil.move(file_path, os.path.join(sorted_dirs[category], filename))
        bucket.append(filename)

    return obfuscated_files, non_obfuscated_files, unresolved_files


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running this script.")

    obfuscated, non_obfuscated, unresolved = resolve_unidentified(DEFAULT_DATASET, api_key)
    print(f"Obfuscated: {len(obfuscated)}, NonObfuscated: {len(non_obfuscated)}, Unresolved: {len(unresolved)}")


if __name__ == "__main__":
    main()
