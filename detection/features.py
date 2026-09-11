"""Shared lexical/statistical feature extraction for JavaScript obfuscation
detection. Used by both the RandomForest and XGBoost pipelines.
"""
import json
import math
import os
import re
import zlib
from collections import Counter

import nltk
from nltk.corpus import words

# Anchored to this file's directory (not the process cwd) so dataset/model
# resolution works the same whether the app is launched via
# `uvicorn detection.main:app` from the repo root or `python main.py` from
# inside detection/.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, "data")


def list_datasets(split="train", data_dir=DATA_DIR):
    """Lists dataset names available under data/<split>/ (subfolder names)."""
    split_root = os.path.join(data_dir, split)
    if not os.path.isdir(split_root):
        return []
    return sorted(
        name for name in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, name))
    )


def resolve_train_dirs(dataset, data_dir=DATA_DIR):
    """Maps a dataset name to its training category directories:
    data/train/<dataset>/{obfuscated,non_obfuscated}."""
    base = os.path.join(data_dir, "train", dataset)
    return {
        "obfuscated": os.path.join(base, "obfuscated"),
        "non_obfuscated": os.path.join(base, "non_obfuscated"),
    }


def resolve_test_dirs(dataset, data_dir=DATA_DIR):
    """Maps a dataset name to its test category directories:
    data/test/<dataset>/{obfuscated,non_obfuscated}."""
    base = os.path.join(data_dir, "test", dataset)
    return {
        "obfuscated": os.path.join(base, "obfuscated"),
        "non_obfuscated": os.path.join(base, "non_obfuscated"),
    }


def resolve_unsorted_dir(dataset, data_dir=DATA_DIR):
    """Maps a dataset name to its unsorted directory: data/unsorted/<dataset>/
    (a flat folder of .js files not yet split into categories)."""
    return os.path.join(data_dir, "unsorted", dataset)


def resolve_sorted_dirs(dataset, data_dir=DATA_DIR):
    """Maps a dataset name to its sorted output directories:
    data/sorted/<dataset>/{obfuscated,non_obfuscated,unidentified}."""
    base = os.path.join(data_dir, "sorted", dataset)
    return {
        "obfuscated": os.path.join(base, "obfuscated"),
        "non_obfuscated": os.path.join(base, "non_obfuscated"),
        "unidentified": os.path.join(base, "unidentified"),
    }


def list_models(algo, package_dir=PACKAGE_DIR):
    """Lists saved model names under models/<algo>/ (filename stems, without
    extension). Neural-network scaler files (<name>_scaler.pkl) are excluded
    so each trained model is listed once regardless of how many files back it."""
    algo_dir = os.path.join(package_dir, "models", algo)
    if not os.path.isdir(algo_dir):
        return []
    names = []
    for filename in os.listdir(algo_dir):
        if filename.endswith('_scaler.pkl'):
            continue
        stem, ext = os.path.splitext(filename)
        if ext in ('.pkl', '.h5'):
            names.append(stem)
    return sorted(names)


FEATURE_COLUMNS = [
    'entropy', 'long_lines_share', 'avg_line_length', 'eval_calls_frequency',
    'string_long_count', 'keyword_freq', 'var_uniqueness', 'token_freq',
    'escape_seq_density', 'english_word_prop', 'compression_ratio',
    'control_flow_count',
]

JS_KEYWORDS = {
    'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else', 'enum',
    'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'null',
    'return', 'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield',
    'let', 'static', 'implements', 'interface', 'package', 'private', 'protected', 'public', 'await', 'abstract',
    'boolean', 'byte', 'char', 'double', 'final', 'float', 'goto', 'int', 'long', 'native', 'short', 'synchronized',
    'throws', 'transient', 'volatile',
}

SUSPICIOUS_TOKENS = ['eval', 'setTimeout', 'Function', 'unescape', 'document.write']


def load_english_words():
    try:
        nltk.data.find('corpora/words.zip')
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    return set(words.words())


ENGLISH_WORDS = load_english_words()


def calculate_entropy(text):
    """Shannon entropy of the text."""
    frequency = Counter(text)
    text_length = len(text)
    return -sum((count / text_length) * math.log2(count / text_length) for count in frequency.values())


def calculate_long_lines(text):
    """Share of lines longer than 1000 characters."""
    lines = text.splitlines()
    if not lines:
        return 0
    long_lines_count = sum(1 for line in lines if len(line) > 1000)
    return long_lines_count / len(lines)


def calculate_avg_line_length(text):
    lines = text.splitlines()
    if not lines:
        return 0
    return sum(len(line) for line in lines) / len(lines)


def eval_frequency(text, total_chars):
    eval_count = text.count('eval')
    return eval_count / total_chars if total_chars else 0


def analyze_strings(text):
    """Counts string literals longer than 50 characters."""
    strings = re.findall(r'["\'](.*?)["\']', text)
    return sum(1 for s in strings if len(s) > 50)


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


def calculate_escape_sequence_density(text):
    r"""Density of escape sequences (e.g., \x, \u) in the code."""
    if not text:
        return 0
    escape_sequences = re.findall(r'(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})', text)
    return len(escape_sequences) / len(text)


def calculate_english_word_proportion(text):
    """Proportion of identifiers that are common English words."""
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
    if not identifiers:
        return 0
    english_count = sum(1 for identifier in identifiers if identifier.lower() in ENGLISH_WORDS)
    return english_count / len(identifiers)


def calculate_compression_ratio(text):
    original_size = len(text)
    if original_size == 0:
        return 0
    compressed_size = len(zlib.compress(text.encode('utf-8')))
    return compressed_size / original_size


def control_flow_complexity(text):
    """Counts occurrences of control-flow keywords."""
    control_flow_keywords = ['if', 'else', 'switch', 'for', 'while', 'do']
    return sum(len(re.findall(r'\b' + keyword + r'\b', text)) for keyword in control_flow_keywords)


def is_only_code_blocks(text):
    """Detects scripts that are only JSON-like objects, structured arrays, or
    object assignments (i.e. not executable/obfuscated code)."""
    text = text.strip()

    js_keywords_exclusion = {
        "function", "var", "let", "const", "for", "if", "while", "switch", "new", "return",
        "eval", "setTimeout", "setInterval", "Function", "unescape", "document.write",
    }
    if any(keyword in text for keyword in js_keywords_exclusion):
        return False

    lines = text.splitlines()
    if all(line.strip().startswith('//') for line in lines) or re.match(r'/\*.*?\*/', text, re.DOTALL):
        return False

    escape_sequences = re.findall(r'(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})', text)
    if len(text) and len(escape_sequences) / len(text) > 0.05:
        return False

    if re.match(r'^\s*\w+(\.\w+)*\s*=\s*{', text) and (text.endswith('};') or text.endswith('}')):
        return True

    if re.match(r'^\s*\w+\s*=\s*\w+\.assign\s*\(\s*\w+\s*\|\|\s*{},\s*{', text) and text.endswith('});'):
        return True

    if re.match(r'^\s*\w+\s*\(\s*\[\s*{', text) and text.endswith('}]\);'):
        return True

    if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
        try:
            json_text = text.replace("'", "\"").replace(r'\/', '/')
            json.loads(json_text)
            return True
        except json.JSONDecodeError:
            pass

    if re.match(r'^\s*<script[^>]*>\s*/\* <!\[CDATA\[\ */\s*{', text, re.DOTALL) and \
            re.search(r'/\* \]\]>\ */\s*</script>\s*$', text, re.DOTALL):
        content = re.search(r'/\* <!\[CDATA\[\ */(.*?)/\* \]\]>\ */', text, re.DOTALL).group(1).strip()
        if re.match(r'^\s*\w+(\.\w+)*\s*=\s*{', content):
            return True

    if len(lines) <= 5:
        block_pattern = re.compile(r'^\s*{[^}]*}\s*$', re.DOTALL)
        is_json_lines = all(
            block_pattern.match(line.strip()) or (line.strip().startswith('[') and line.strip().endswith(']'))
            for line in lines
        )
        if is_json_lines:
            return True

    if all(line.strip().startswith('{') and line.strip().endswith('}') for line in lines):
        return True

    suspicious_patterns = [
        r'\bwhile\s*\(',
        r'\bfor\s*\(',
        r'eval\s*\(',
        r'setTimeout\s*\(',
        r'Function\s*\(',
        r'\bdocument\.write\b',
    ]
    if any(re.search(pattern, text) for pattern in suspicious_patterns):
        return False

    return False


def analyze_file(file_path):
    """Reads a file and calculates its feature vector, plus whether it is
    structurally non-obfuscated data (JSON/objects) rather than executable
    code. The structural flag - not the model - decides the label/prediction
    for such files (a JSON blob can't be "obfuscated" by definition,
    regardless of which folder it was filed under or what the heuristics
    say), but the real feature vector is still returned so training data
    never contains a fabricated all-zero row."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read().strip()

    total_chars = len(text)

    metrics = (
        calculate_entropy(text),
        calculate_long_lines(text),
        calculate_avg_line_length(text),
        eval_frequency(text, total_chars),
        analyze_strings(text),
        keyword_frequency(text),
        variable_uniqueness(text),
        detect_token_frequency(text),
        calculate_escape_sequence_density(text),
        calculate_english_word_proportion(text),
        calculate_compression_ratio(text),
        control_flow_complexity(text),
    )
    return metrics, is_only_code_blocks(text)


def extract_features_with_filenames(directory, label, comment=None):
    """Extracts features and filenames from all .js files in a directory.
    Files that are structurally non-obfuscated data (JSON/objects) are
    always labeled NonObfuscated regardless of which folder they came from,
    but still carry their real computed feature vector."""
    features = []
    filenames = []
    comments = []
    for filename in os.listdir(directory):
        if not filename.endswith('.js'):
            continue
        file_path = os.path.join(directory, filename)
        metrics, is_structural = analyze_file(file_path)
        effective_label = 0 if is_structural else label
        features.append(list(metrics) + [effective_label])
        comments.append("Detected as NonObfuscated by structure" if is_structural else comment)
        filenames.append(filename)
    return features, filenames, comments


def extract_features_for_prediction(directory):
    """Extracts features and filenames from every .js file in a directory,
    for prediction (no ground-truth label needed). Also returns
    forced_labels: 0 for files that are structurally non-obfuscated data
    (JSON/objects), so callers can skip asking the model for those; None
    otherwise."""
    feature_rows = []
    filenames = []
    forced_labels = []
    for filename in os.listdir(directory):
        if not filename.endswith('.js'):
            continue
        file_path = os.path.join(directory, filename)
        metrics, is_structural = analyze_file(file_path)
        feature_rows.append(list(metrics))
        forced_labels.append(0 if is_structural else None)
        filenames.append(filename)
    return feature_rows, filenames, forced_labels
