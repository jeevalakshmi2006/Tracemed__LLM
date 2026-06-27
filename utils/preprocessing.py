import csv
import math
import random

# ─────────────────────────────────────────
#  LOAD CSV
# ─────────────────────────────────────────
def load_csv(filepath):
    dataset = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    return dataset


# ─────────────────────────────────────────
#  CLEAN & CONVERT
# ─────────────────────────────────────────
def clean_data(dataset):
    cleaned = []
    for row in dataset:
        try:
            clean_row = {
                'age':      float(row['age']),
                'sex':      float(row['sex']),
                'cp':       float(row['cp']),
                'trestbps': float(row['trestbps']),
                'chol':     float(row['chol']),
                'fbs':      float(row['fbs']),
                'restecg':  float(row['restecg']),
                'thalach':  float(row['thalach']),
                'exang':    float(row['exang']),
                'oldpeak':  float(row['oldpeak']),
                'slope':    float(row['slope']),
                'ca':       float(row['ca']),
                'thal':     float(row['thal']),
                'target':   1 if float(row['target']) > 0 else 0  # binary: 0 or 1
            }
            cleaned.append(clean_row)
        except ValueError:
            pass  # skip rows with missing/invalid values
    return cleaned


# ─────────────────────────────────────────
#  NORMALIZE (Min-Max Scaling)
#  scales every feature to 0.0 – 1.0
# ─────────────────────────────────────────
FEATURE_COLS = ['age','sex','cp','trestbps','chol','fbs',
                'restecg','thalach','exang','oldpeak','slope','ca','thal']

def compute_min_max(dataset):
    stats = {}
    for col in FEATURE_COLS:
        values = [row[col] for row in dataset]
        stats[col] = {'min': min(values), 'max': max(values)}
    return stats

def normalize(dataset, stats):
    normalized = []
    for row in dataset:
        norm_row = {}
        for col in FEATURE_COLS:
            mn = stats[col]['min']
            mx = stats[col]['max']
            if mx - mn == 0:
                norm_row[col] = 0.0
            else:
                norm_row[col] = (row[col] - mn) / (mx - mn)
        norm_row['target'] = row['target']
        normalized.append(norm_row)
    return normalized


# ─────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────
def train_test_split(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    data = dataset[:]
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


# ─────────────────────────────────────────
#  ROW → FEATURE LIST  (for tree input)
# ─────────────────────────────────────────
def row_to_features(row):
    return [row[col] for col in FEATURE_COLS]

def row_to_label(row):
    return row['target']


# ─────────────────────────────────────────
#  FULL PIPELINE  (call this from main.py)
# ─────────────────────────────────────────
def prepare_data(filepath):
    raw        = load_csv(filepath)
    cleaned    = clean_data(raw)
    stats      = compute_min_max(cleaned)
    normalized = normalize(cleaned, stats)
    train, test = train_test_split(normalized)

    X_train = [row_to_features(r) for r in train]
    y_train = [row_to_label(r)    for r in train]
    X_test  = [row_to_features(r) for r in test]
    y_test  = [row_to_label(r)    for r in test]

    print(f"Dataset loaded   : {len(cleaned)} rows")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")

    return X_train, y_train, X_test, y_test, stats