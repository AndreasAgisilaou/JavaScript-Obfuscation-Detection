"""Trains a small feed-forward neural network to detect obfuscated JavaScript
files based on hand-crafted lexical/statistical heuristics, and saves it to
detection/models/neural_network/<model_name>.h5 (+ a matching scaler .pkl).
"""
import base64
import binascii
import os
import math
import re
import zlib
from collections import Counter

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

try:
    from detection.features import PACKAGE_DIR, resolve_train_dirs, resolve_test_dirs
except ImportError:
    # Allows running this file directly (e.g. `python naural_network.py` from
    # inside detection/), where "detection" isn't an importable package.
    from features import PACKAGE_DIR, resolve_train_dirs, resolve_test_dirs

# --- Configuration ---
DEFAULT_DATASET = "dataset_v1"
DEFAULT_MODEL_NAME = "neural_network_model"

MODELS_DIR = os.path.join(PACKAGE_DIR, "models", "neural_network")

FEATURE_COLUMNS = [
    'entropy', 'comment_density', 'avg_line_length', 'longest_line',
    'keyword_frequency', 'variable_uniqueness', 'token_frequency',
    'numeric_density', 'english_word_proportion', 'compression_ratio',
    'hex_literal_density', 'encoding_patterns', 'function_call_density',
]

JS_KEYWORDS = {
    'break', 'case', 'catch', 'continue', 'debugger', 'default', 'delete',
    'do', 'else', 'finally', 'for', 'function', 'if', 'in', 'instanceof',
    'new', 'return', 'switch', 'this', 'throw', 'try', 'typeof', 'var',
    'void', 'while', 'with', 'let', 'const', 'class', 'extends', 'super',
    'import', 'export', 'yield',
}

COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
}

SUSPICIOUS_TOKENS = ['eval', 'setTimeout', 'Function', 'unescape', 'document.write']


# --- Feature extraction ---

def calculate_entropy(text):
    """Shannon entropy of the text."""
    frequency = Counter(text)
    text_length = len(text)
    return -sum((count / text_length) * math.log2(count / text_length) for count in frequency.values())


def calculate_comment_density(text):
    lines = text.splitlines()
    if not lines:
        return 0
    comment_lines = sum(1 for line in lines if line.strip().startswith('//') or line.strip().startswith('/*'))
    return comment_lines / len(lines)


def calculate_avg_line_length(text):
    lines = text.splitlines()
    if not lines:
        return 0
    return sum(len(line) for line in lines) / len(lines)


def calculate_longest_line(text):
    lines = text.splitlines()
    if not lines:
        return 0
    return max(len(line) for line in lines)


def keyword_frequency(text):
    words_list = re.findall(r'\b\w+\b', text)
    if not words_list:
        return 0
    keyword_count = sum(1 for word in words_list if word in JS_KEYWORDS)
    return keyword_count / len(words_list)


def variable_uniqueness(text):
    """Ratio of unique identifier names to total identifier occurrences."""
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    variables = [identifier for identifier in identifiers if identifier not in JS_KEYWORDS]
    if not variables:
        return 0
    return len(set(variables)) / len(variables)


def detect_token_frequency(text):
    """Counts occurrences of suspicious JavaScript tokens."""
    return sum(text.count(token) for token in SUSPICIOUS_TOKENS)


def calculate_numeric_density(text):
    numbers = re.findall(r'\b\d+\b', text)
    total_tokens = len(re.findall(r'\b\w+\b', text))
    return len(numbers) / total_tokens if total_tokens else 0


def calculate_english_word_proportion(text):
    """Proportion of identifiers that are common English words."""
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    if not identifiers:
        return 0
    english_count = sum(1 for identifier in identifiers if identifier.lower() in COMMON_ENGLISH_WORDS)
    return english_count / len(identifiers)


def calculate_compression_ratio(text):
    original_size = len(text)
    if original_size == 0:
        return 0
    compressed_size = len(zlib.compress(text.encode('utf-8')))
    return compressed_size / original_size


def calculate_hex_literal_density(text):
    """Density of hexadecimal literals (e.g., 0xFF) in the text."""
    if not text:
        return 0
    hex_literals = re.findall(r'0x[0-9a-fA-F]+', text)
    return len(hex_literals) / len(text)


def _looks_like_base64(candidate):
    """True if candidate decodes cleanly as base64 and contains at least one
    digit or '+'/'/' - plain runs of letters (long identifiers, camelCase
    chains, English words) pad-decode "successfully" too, so length and
    valid-alphabet alone aren't enough to tell them apart from real
    base64-encoded data."""
    if not any(c.isdigit() or c in '+/' for c in candidate):
        return False
    padded = candidate + '=' * (-len(candidate) % 4)
    try:
        base64.b64decode(padded, validate=True)
    except (ValueError, binascii.Error):
        return False
    return True


def detect_encoding_patterns(text):
    """Counts encoded strings such as hexadecimal, Unicode, or Base64 patterns."""
    hex_count = len(re.findall(r'\\x[0-9a-fA-F]{2}', text))
    unicode_count = len(re.findall(r'\\u[0-9a-fA-F]{4}', text))
    base64_candidates = re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', text)
    base64_count = sum(1 for candidate in base64_candidates if _looks_like_base64(candidate))
    return hex_count + unicode_count + base64_count


def calculate_function_call_density(text):
    function_calls = re.findall(r'\b\w+\s*\(', text)
    total_tokens = len(re.findall(r'\b\w+\b', text))
    return len(function_calls) / total_tokens if total_tokens else 0


def analyze_file(file_path, label=None):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read().strip()

    return [
        calculate_entropy(text),
        calculate_comment_density(text),
        calculate_avg_line_length(text),
        calculate_longest_line(text),
        keyword_frequency(text),
        variable_uniqueness(text),
        detect_token_frequency(text),
        calculate_numeric_density(text),
        calculate_english_word_proportion(text),
        calculate_compression_ratio(text),
        calculate_hex_literal_density(text),
        detect_encoding_patterns(text),
        calculate_function_call_density(text),
        label,
    ]


def extract_features_with_filenames(directory, label):
    features = []
    filenames = []
    for filename in os.listdir(directory):
        if not filename.endswith('.js'):
            continue
        file_path = os.path.join(directory, filename)
        features.append(analyze_file(file_path, label))
        filenames.append(filename)
    return features, filenames


def build_dataframe(obfuscated_dir, nonobfuscated_dir):
    obfuscated, _ = extract_features_with_filenames(obfuscated_dir, label=1)
    non_obfuscated, _ = extract_features_with_filenames(nonobfuscated_dir, label=0)

    all_features = obfuscated + non_obfuscated
    return pd.DataFrame(all_features, columns=FEATURE_COLUMNS + ['label'])


def extract_features_for_prediction(directory):
    """Extracts features and filenames from every .js file in a directory,
    for prediction (no ground-truth label needed)."""
    feature_rows = []
    filenames = []
    for filename in os.listdir(directory):
        if not filename.endswith('.js'):
            continue
        file_path = os.path.join(directory, filename)
        feature_rows.append(analyze_file(file_path)[:-1])  # drop the trailing (unset) label slot
        filenames.append(filename)
    return feature_rows, filenames


def build_model(input_dim):
    # Sized for a 13-feature tabular input, not image/text-scale data - a
    # much wider network here mostly adds overfitting capacity rather than
    # learning power, especially with EarlyStopping/ReduceLROnPlateau
    # watching a small validation split.
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(2, activation='softmax'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(dataset, model_name, epochs=50, batch_size=32):
    """Trains on data/train/<dataset>/ and saves the trained Keras model to
    models/neural_network/<model_name>.h5, and its scaler to
    models/neural_network/<model_name>_scaler.pkl."""
    train_dirs = resolve_train_dirs(dataset)
    df_train = build_dataframe(train_dirs["obfuscated"], train_dirs["non_obfuscated"])
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encoded = to_categorical(y_train, num_classes=2)

    model = build_model(X_train_scaled.shape[1])

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    model.fit(
        X_train_scaled, y_train_encoded,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.2,
        callbacks=[lr_scheduler, early_stopping],
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path


def evaluate_model(model_path, scaler_path, test_dataset):
    """Scores a saved model against data/test/<test_dataset>/, returning
    accuracy as the fraction of files correctly classified (0-1)."""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    test_dirs = resolve_test_dirs(test_dataset)
    df_test = build_dataframe(test_dirs["obfuscated"], test_dirs["non_obfuscated"])
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled).argmax(axis=1)
    return accuracy_score(y_test, predictions)


def predict_directory(model_path, scaler_path, directory):
    """Predicts every .js file in a directory using a saved model. Returns
    {filename: 1|0} (1 = obfuscated, 0 = non_obfuscated)."""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    feature_rows, filenames = extract_features_for_prediction(directory)
    if not filenames:
        return {}

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled).argmax(axis=1)
    return dict(zip(filenames, (int(p) for p in predictions)))


def main():
    model_path, scaler_path = train_model(DEFAULT_DATASET, DEFAULT_MODEL_NAME)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()
