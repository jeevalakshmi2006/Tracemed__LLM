from utils.preprocessing import prepare_data
from utils.decision_tree import build_tree_with_proba, predict_one, accuracy
from utils.random_forest import RandomForest, evaluate
from utils.feature_importance import (
    permutation_importance, normalize_importances,
    get_top_features, patient_explanation, print_explanation
)
from utils.llm_explanation import get_clinical_explanation, get_patient_explanation

X_train, y_train, X_test, y_test, stats = prepare_data('data/heart.csv')
print("First training row:", X_train[0])
print("First label:", y_train[0])
print("Building decision tree...")
tree = build_tree_with_proba(X_train, y_train, max_depth=6, min_samples=5)

acc = accuracy(tree, X_test, y_test)
print(f"Decision Tree Accuracy: {acc * 100:.2f}%")

# Train Random Forest
forest = RandomForest(n_trees=50, max_depth=8, min_samples=3)
forest.fit(X_train, y_train)

# Evaluate
evaluate(forest, X_test, y_test)

# Test on one patient
sample = X_test[0]
label, proba = forest.predict_one_row(sample)
print(f"Sample Patient → Risk: {proba*100:.1f}% | Prediction: {'Heart Disease' if label==1 else 'No Disease'}")
# Global feature importance
print("\nCalculating feature importance...")
importances, baseline = permutation_importance(forest, X_test, y_test, n_repeats=5)
norm = normalize_importances(importances)
top  = get_top_features(norm, top_n=5)

print("\n── Top 5 Important Features ────────────")
for f in top:
    print(f"  {f['feature_name']:<22} {f['importance']}%")

# Patient-level explanation
print("\nExplaining sample patient...")
sample     = X_test[0]
contribs, risk = patient_explanation(sample, X_train, forest, top_n=5)
print_explanation(contribs, risk)
# ── LLM Explanations ─────────────────────
print("\nGenerating doctor explanation...")
doctor_text = get_clinical_explanation(risk, contribs, sample)
print("\n── Doctor Explanation ──────────────────")
print(doctor_text)
 
print("\nGenerating patient explanation...")
patient_text = get_patient_explanation(risk, contribs, sample)
print("\n── Patient Explanation ─────────────────")
print(patient_text)