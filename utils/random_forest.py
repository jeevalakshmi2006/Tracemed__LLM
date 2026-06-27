import random
import math
from utils.decision_tree import build_tree_with_proba, predict_one, predict_proba_one

# ─────────────────────────────────────────
#  BOOTSTRAP SAMPLING
#  randomly pick N rows WITH replacement
#  this is what makes each tree different
# ─────────────────────────────────────────
def bootstrap_sample(X, y, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(X)
    indices = [random.randint(0, n - 1) for _ in range(n)]
    sample_X = [X[i] for i in indices]
    sample_y = [y[i] for i in indices]
    return sample_X, sample_y


# ─────────────────────────────────────────
#  RANDOM FOREST CLASS
# ─────────────────────────────────────────
class RandomForest:
    def __init__(self, n_trees=50, max_depth=8, min_samples=3, n_features=None):
        self.n_trees    = n_trees       # how many trees to build
        self.max_depth  = max_depth     # max depth of each tree
        self.min_samples = min_samples  # min samples to split a node
        self.n_features = n_features    # features per split (None = sqrt of total)
        self.trees      = []            # list of trained trees

    def fit(self, X, y):
        self.trees = []

        # auto-set n_features to sqrt of total features if not given
        if self.n_features is None:
            self.n_features = max(1, int(math.sqrt(len(X[0]))))

        print(f"Training Random Forest: {self.n_trees} trees | "
              f"max_depth={self.max_depth} | "
              f"features_per_split={self.n_features}")

        for i in range(self.n_trees):
            # 1. bootstrap sample
            bX, by = bootstrap_sample(X, y, seed=i)

            # 2. build one tree on that sample
            tree = build_tree_with_proba(
                bX, by,
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                n_features=self.n_features
            )

            self.trees.append(tree)

            # progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Trees built: {i + 1}/{self.n_trees}")

        print("Random Forest training complete.")

    # ── PREDICT CLASS (0 or 1) ───────────
    def predict(self, X):
        predictions = []
        for row in X:
            votes = [predict_one(tree, row) for tree in self.trees]
            result = 1 if sum(votes) >= len(votes) / 2 else 0
            predictions.append(result)
        return predictions

    # ── PREDICT PROBABILITY (0.0–1.0) ────
    def predict_proba(self, X):
        probas = []
        for row in X:
            tree_probas = [predict_proba_one(tree, row) for tree in self.trees]
            avg_proba = sum(tree_probas) / len(tree_probas)
            probas.append(avg_proba)
        return probas

    # ── PREDICT SINGLE ROW ───────────────
    def predict_one_row(self, row):
        votes = [predict_one(tree, row) for tree in self.trees]
        proba = sum(predict_proba_one(tree, row) for tree in self.trees) / len(self.trees)
        label = 1 if sum(votes) >= len(votes) / 2 else 0
        return label, proba


# ─────────────────────────────────────────
#  EVALUATION METRICS  (no libraries)
# ─────────────────────────────────────────
def evaluate(forest, X_test, y_test):
    preds = forest.predict(X_test)

    tp = sum(1 for p, a in zip(preds, y_test) if p == 1 and a == 1)
    tn = sum(1 for p, a in zip(preds, y_test) if p == 0 and a == 0)
    fp = sum(1 for p, a in zip(preds, y_test) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(preds, y_test) if p == 0 and a == 1)

    accuracy  = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print("\n── Evaluation Results ──────────────────")
    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall    * 100:.2f}%")
    print(f"  F1 Score  : {f1        * 100:.2f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print("────────────────────────────────────────\n")

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }