import numpy as np
# If split value cannot split the thing (if one is empty) then call it a leaf and don't split further

class Node:
    def __init__(self, value):
        self.value = value  # tuple: [split_feature, split_value, left_prediction, right_prediction]
        self.left = None
        self.right = None


class Tree:
    def __init__(self):
        self.root = None  # Starts empty
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            raise ValueError("Node already exists, can only insert at 'root'")
    def predict(self, x):
        def traverse_nodes(active_node, inpu) :
            if active_node.value[0] is None :
                return 0.5*(active_node.value[2] + active_node.value[3]) # POSSIBLY BAD. CHECK IF ROBUST.
            else :
                sp_ft = active_node.value[0]
                sp_val = active_node.value[1]
                if inpu[sp_ft] <= sp_val:
                    return traverse_nodes(active_node.left, inpu)
                else :
                    return traverse_nodes(active_node.right, inpu)

        if self.root is None :
            raise ValueError("Node empty")
        else :
            return traverse_nodes(self.root, x)


# -------------------------------------------------------------------------------------------------

def mean_prediction(y: np.ndarray):
    return np.mean(y)


def branch_error(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    mean = mean_prediction(y)
    return np.sum((y - mean) ** 2)


def make_error_array(x: np.ndarray, y: np.ndarray):
    """
    x shape: (num_features, num_samples)
    y shape: (num_samples,wtf)

    Returns:
        split_feature (int)    : index of the best feature to split on
        split_value   (float)  : the threshold value to split at
        left_x, left_y         : samples where x[split_feature] <= split_value
        right_x, right_y       : samples where x[split_feature] >  split_value
    """
    num_features, num_samples = x.shape

    # Build a ragged structure: for each feature, store its candidate breakpoints.
    # feature_breaks[f]       -> 1D array of breakpoints for feature f
    # feature_break_errors[f] -> 1D array of total errors for each breakpoint of feature f
    feature_breaks = []
    feature_break_errors = []

    for f in range(num_features):
        feature_vals = x[f]                          # shape: (num_samples,)
        sorted_vals  = np.sort(feature_vals)

        # midpoints between consecutive unique sorted values
        breaks = 0.5 * (sorted_vals[:-1] + sorted_vals[1:])  # shape: (num_samples-1,)

        errors = np.empty(len(breaks))
        for b_idx, bp in enumerate(breaks):
            mask        = feature_vals <= bp
            left_error  = branch_error(y[mask])
            right_error = branch_error(y[~mask])
            errors[b_idx] = left_error + right_error

        feature_breaks.append(breaks)
        feature_break_errors.append(errors)

    # Find the (feature, breakpoint) pair with the lowest total error
    best_error   = np.inf
    split_feature = 0
    split_b_idx   = 0

    for f in range(num_features):
        local_min_idx = np.argmin(feature_break_errors[f])
        local_min     = feature_break_errors[f][local_min_idx]
        if local_min < best_error:
            best_error    = local_min
            split_feature = f
            split_b_idx   = local_min_idx

    split_value = feature_breaks[split_feature][split_b_idx]

    # Partition x and y on the best split
    mask        = x[split_feature] <= split_value
    left_x,  left_y  = x[:, mask],  y[mask]
    right_x, right_y = x[:, ~mask], y[~mask]

    return split_feature, split_value, left_x, left_y, right_x, right_y


def build_tree(x: np.ndarray, y: np.ndarray, smallest_class: int = 1) -> Tree:
    """
    Recursively builds a decision tree.

    Each internal node stores:  [split_feature, split_value, None,            None           ]
    Each leaf node stores:      [None,          None,        left_prediction,  right_prediction]

    x shape: (num_features, num_samples)
    y shape: (num_samples,wtf)
    """
    DT = Tree()

    if y.shape == () :
        return DT

    # Stopping condition: too few samples → leaf node
    if x.shape[1] <= smallest_class:
        prediction = mean_prediction(y)
        # Leaf: split fields unused; both predictions hold the same constant estimate
        DT.insert(np.array([None, None, prediction, prediction]))
        return DT

    # Find the best split
    split_feature, split_value, left_x, left_y, right_x, right_y = make_error_array(x, y)
    print(f"sf, sv, lx, rx, = {split_feature}, {split_value}, {left_x}, {right_x}")
    # Internal node: prediction fields unused at this level
    DT.insert(np.array([split_feature, split_value, None, None], dtype=object))
    if left_x.shape[1] == 0 or right_x.shape[1] == 0 :
        print(f"False split! Will make leaf.")
        prediction = mean_prediction(y)
        DT.insert(np.array([None, None, prediction, prediction]))
        return DT
    else :
        # Recurse into left and right subtrees
        left_subtree  = build_tree(left_x,  left_y,  smallest_class)
        right_subtree = build_tree(right_x, right_y, smallest_class)

        DT.root.left  = left_subtree.root
        DT.root.right = right_subtree.root

    return DT