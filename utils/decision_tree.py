import random

# ─────────────────────────────────────────
#  GINI IMPURITY
#  measures how "mixed" a group of labels is
#  0.0 = perfectly pure, 0.5 = perfectly mixed
# ─────────────────────────────────────────
def gini_impurity(labels):
    if not labels:
        return 0.0
    total = len(labels)
    count_1 = sum(labels)
    count_0 = total - count_1
    p1 = count_1 / total
    p0 = count_0 / total
    return 1.0 - (p0 ** 2 + p1 ** 2)


# ─────────────────────────────────────────
#  WEIGHTED GINI after a split
# ─────────────────────────────────────────
def weighted_gini(left_labels, right_labels):
    total = len(left_labels) + len(right_labels)
    if total == 0:
        return 0.0
    left_weight  = len(left_labels)  / total
    right_weight = len(right_labels) / total
    return (left_weight  * gini_impurity(left_labels) +
            right_weight * gini_impurity(right_labels))


# ─────────────────────────────────────────
#  SPLIT DATASET on a feature and threshold
# ─────────────────────────────────────────
def split_dataset(X, y, feature_index, threshold):
    left_X,  left_y  = [], []
    right_X, right_y = [], []
    for row, label in zip(X, y):
        if row[feature_index] <= threshold:
            left_X.append(row)
            left_y.append(label)
        else:
            right_X.append(row)
            right_y.append(label)
    return left_X, left_y, right_X, right_y


# ─────────────────────────────────────────
#  FIND BEST SPLIT
#  tries every feature and threshold
#  returns the one with lowest gini
# ─────────────────────────────────────────
def best_split(X, y, n_features=None):
    total_features = len(X[0])

    # if n_features given, randomly pick that many (used by Random Forest)
    if n_features is None:
        feature_indices = list(range(total_features))
    else:
        feature_indices = random.sample(range(total_features),
                                        min(n_features, total_features))

    best_gini      = float('inf')
    best_index     = None
    best_threshold = None
    best_splits    = None

    for fi in feature_indices:
        # collect unique values in this feature as candidate thresholds
        thresholds = sorted(set(row[fi] for row in X))

        for threshold in thresholds:
            lX, ly, rX, ry = split_dataset(X, y, fi, threshold)

            if not ly or not ry:
                continue  # skip useless splits

            gini = weighted_gini(ly, ry)

            if gini < best_gini:
                best_gini      = gini
                best_index     = fi
                best_threshold = threshold
                best_splits    = (lX, ly, rX, ry)

    return best_index, best_threshold, best_splits, best_gini


# ─────────────────────────────────────────
#  MAJORITY VOTE  (leaf prediction)
# ─────────────────────────────────────────
def majority_vote(labels):
    return 1 if sum(labels) >= len(labels) / 2 else 0


# ─────────────────────────────────────────
#  NODE CLASS
#  represents one node in the tree
# ─────────────────────────────────────────
class Node:
    def __init__(self):
        self.feature_index = None   # which feature to split on
        self.threshold     = None   # split value
        self.left          = None   # left child Node
        self.right         = None   # right child Node
        self.is_leaf       = False
        self.prediction    = None   # only set if leaf


# ─────────────────────────────────────────
#  BUILD TREE  (recursive)
# ─────────────────────────────────────────
def build_tree(X, y, max_depth=10, min_samples=2, n_features=None, depth=0):
    node = Node()

    # ── STOP CONDITIONS ──────────────────
    # 1. all labels are the same
    if len(set(y)) == 1:
        node.is_leaf    = True
        node.prediction = y[0]
        return node

    # 2. too few samples
    if len(y) < min_samples:
        node.is_leaf    = True
        node.prediction = majority_vote(y)
        return node

    # 3. max depth reached
    if depth >= max_depth:
        node.is_leaf    = True
        node.prediction = majority_vote(y)
        return node

    # ── FIND BEST SPLIT ──────────────────
    fi, threshold, splits, gini = best_split(X, y, n_features)

    # no useful split found
    if fi is None:
        node.is_leaf    = True
        node.prediction = majority_vote(y)
        return node

    lX, ly, rX, ry = splits

    # ── SET NODE VALUES ──────────────────
    node.feature_index = fi
    node.threshold     = threshold

    # ── RECURSE ──────────────────────────
    node.left  = build_tree(lX, ly, max_depth, min_samples, n_features, depth + 1)
    node.right = build_tree(rX, ry, max_depth, min_samples, n_features, depth + 1)

    return node


# ─────────────────────────────────────────
#  PREDICT ONE SAMPLE
# ─────────────────────────────────────────
def predict_one(node, row):
    if node.is_leaf:
        return node.prediction
    if row[node.feature_index] <= node.threshold:
        return predict_one(node.left, row)
    else:
        return predict_one(node.right, row)


# ─────────────────────────────────────────
#  PREDICT PROBABILITY  (% of leaf labels)
#  We store leaf label counts for this
# ─────────────────────────────────────────
def build_tree_with_proba(X, y, max_depth=10, min_samples=2, n_features=None, depth=0):
    node = Node()

    if len(set(y)) == 1:
        node.is_leaf    = True
        node.prediction = y[0]
        node.proba      = float(y[0])
        return node

    if len(y) < min_samples or depth >= max_depth:
        node.is_leaf    = True
        node.prediction = majority_vote(y)
        node.proba      = sum(y) / len(y)
        return node

    fi, threshold, splits, gini = best_split(X, y, n_features)

    if fi is None:
        node.is_leaf    = True
        node.prediction = majority_vote(y)
        node.proba      = sum(y) / len(y)
        return node

    lX, ly, rX, ry = splits
    node.feature_index = fi
    node.threshold     = threshold
    node.left  = build_tree_with_proba(lX, ly, max_depth, min_samples, n_features, depth + 1)
    node.right = build_tree_with_proba(rX, ry, max_depth, min_samples, n_features, depth + 1)

    return node


def predict_proba_one(node, row):
    if node.is_leaf:
        return node.proba
    if row[node.feature_index] <= node.threshold:
        return predict_proba_one(node.left, row)
    else:
        return predict_proba_one(node.right, row)


# ─────────────────────────────────────────
#  ACCURACY
# ─────────────────────────────────────────
def accuracy(tree, X_test, y_test):
    correct = 0
    for row, label in zip(X_test, y_test):
        if predict_one(tree, row) == label:
            correct += 1
    return correct / len(y_test)