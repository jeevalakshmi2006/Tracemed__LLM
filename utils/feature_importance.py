import random
from utils.decision_tree import predict_proba_one

# ─────────────────────────────────────────
#  FEATURE NAMES  (human readable)
# ─────────────────────────────────────────
FEATURE_NAMES = [
    'Age', 'Sex', 'Chest Pain Type', 'Blood Pressure',
    'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG',
    'Max Heart Rate', 'Exercise Angina', 'ST Depression',
    'Slope', 'Major Vessels', 'Thal'
]

FEATURE_KEYS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# ─────────────────────────────────────────
#  PERMUTATION IMPORTANCE  (from scratch)
#  idea: shuffle one feature at a time
#        measure how much accuracy drops
#        bigger drop = more important feature
# ─────────────────────────────────────────
def permutation_importance(forest, X, y, n_repeats=10, seed=42):
    random.seed(seed)
    n_features = len(X[0])

    # baseline accuracy with all features intact
    preds    = forest.predict(X)
    baseline = sum(1 for p, a in zip(preds, y) if p == a) / len(y)

    importances = []

    for fi in range(n_features):
        drops = []

        for _ in range(n_repeats):
            # copy X and shuffle column fi
            X_permuted = [list(row) for row in X]
            col_values = [row[fi] for row in X_permuted]
            random.shuffle(col_values)
            for i, row in enumerate(X_permuted):
                row[fi] = col_values[i]

            # accuracy with shuffled feature
            preds_p   = forest.predict(X_permuted)
            acc_p     = sum(1 for p, a in zip(preds_p, y) if p == a) / len(y)
            drops.append(baseline - acc_p)

        avg_drop = sum(drops) / len(drops)
        importances.append(avg_drop)

    return importances, baseline


# ─────────────────────────────────────────
#  NORMALIZE IMPORTANCES to percentages
# ─────────────────────────────────────────
def normalize_importances(importances):
    # clip negatives to 0 (feature had no effect)
    clipped = [max(0.0, v) for v in importances]
    total   = sum(clipped)
    if total == 0:
        return [0.0] * len(clipped)
    return [v / total for v in clipped]


# ─────────────────────────────────────────
#  GET TOP N FEATURES  (sorted by impact)
# ─────────────────────────────────────────
def get_top_features(importances_norm, top_n=5):
    ranked = sorted(
        enumerate(importances_norm),
        key=lambda x: x[1],
        reverse=True
    )
    results = []
    for idx, score in ranked[:top_n]:
        results.append({
            'feature_index': idx,
            'feature_name':  FEATURE_NAMES[idx],
            'feature_key':   FEATURE_KEYS[idx],
            'importance':    round(score * 100, 1)   # as percentage
        })
    return results


# ─────────────────────────────────────────
#  PATIENT-LEVEL EXPLANATION
#  compares this patient's feature values
#  against the average of the training set
#  to show which values pushed risk up/down
# ─────────────────────────────────────────
def patient_explanation(patient_row, X_train, forest, top_n=5):
    n_features = len(patient_row)

    # compute column means from training data
    means = []
    for fi in range(n_features):
        col = [row[fi] for row in X_train]
        means.append(sum(col) / len(col))

    # baseline risk using mean values
    baseline_proba = sum(
        predict_proba_one(tree, means) for tree in forest.trees
    ) / len(forest.trees)

    # actual patient risk
    patient_proba = sum(
        predict_proba_one(tree, patient_row) for tree in forest.trees
    ) / len(forest.trees)

    contributions = []
    for fi in range(n_features):
        # build a row = means but replace fi with patient value
        modified = list(means)
        modified[fi] = patient_row[fi]

        modified_proba = sum(
            predict_proba_one(tree, modified) for tree in forest.trees
        ) / len(forest.trees)

        # contribution = change from baseline
        contribution = modified_proba - baseline_proba
        contributions.append({
            'feature_index': fi,
            'feature_name':  FEATURE_NAMES[fi],
            'feature_key':   FEATURE_KEYS[fi],
            'patient_value': patient_row[fi],
            'mean_value':    means[fi],
            'contribution':  round(contribution * 100, 1)
        })

    # sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return contributions[:top_n], round(patient_proba * 100, 1)


# ─────────────────────────────────────────
#  PRINT EXPLANATION  (terminal preview)
# ─────────────────────────────────────────
def print_explanation(contributions, risk_score):
    print(f"\n── Patient Risk Score: {risk_score}% ──────────────")
    print(f"{'Feature':<22} {'Contribution':>12}  Direction")
    print("─" * 50)
    for c in contributions:
        direction = "▲ increases risk" if c['contribution'] > 0 else "▼ decreases risk"
        print(f"{c['feature_name']:<22} {c['contribution']:>+10.1f}%  {direction}")
    print("─" * 50)