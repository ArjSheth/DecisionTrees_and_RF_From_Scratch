import numpy as np
import random as rnd
# If split value cannot split the thing (if one is empty) then call it a leaf and don't split further
# Red marked : Fine.


class Node:
    def __init__(self, value):
        self.value = value  # tuple: [split_feature, split_value, left_prediction, right_prediction]
        self.left = None
        self.right = None

# -------------------------------------------------------------------------------------------------

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
            if active_node.value[0] is None : # meaning split feature is None, i.e, leaf node.
                return active_node.value[2] # POSSIBLY BAD. CHECK IF ROBUST.
            else :
                sp_ft = active_node.value[0]
                sp_val = active_node.value[1]
                if inpu[sp_ft] <= sp_val:
                    return traverse_nodes(active_node.left, inpu)
                else :
                    return traverse_nodes(active_node.right, inpu)
        # ------
        if self.root is None :
            raise ValueError("Node empty")
        else :
            return traverse_nodes(self.root, x)


# -------------------------------------------------------------------------------------------------

def mean_prediction(y: np.ndarray):
    return np.mean(y)

# -------------------------------------------------------------------------------------------------

def branch_error(y: np.ndarray) -> float:
    if y.shape[0] == 0: # corresponds to a shape like () instead of (num_of_samples,).
        return 0.0
    mean = mean_prediction(y)
    return np.sum((y - mean) ** 2)

# -------------------------------------------------------------------------------------------------

def make_error_array(x: np.ndarray, y: np.ndarray):
    """
    x shape: (num_features, num_samples);
    y shape: (num_samples,).

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
        feature_vals = x[f] # shape: (num_samples,)
        interim_sorted_vals  = np.unique(feature_vals) # can contain duplicates.
        sorted_vals = np.sort(interim_sorted_vals) # no duplicates.
        # Now we can use sorted_vals, which may be singleton.
        if len(sorted_vals) == 1 : # if there is only one value taken by y
            breaks = sorted_vals # then break point is just that value
            # At this split value, error will only accumulate for one side.
            errors = np.array([branch_error(y)])
            # ------------------------------------------
            feature_breaks.append(breaks)
            feature_break_errors.append(errors)
        else:
            breaks = 0.5 * (sorted_vals[:-1] + sorted_vals[1:])  # shape: (num_samples-1,)
            errors = np.empty(len(breaks))
            for b_idx, bp in enumerate(breaks):
                mask        = feature_vals <= bp
                left_error  = branch_error(y[mask])
                right_error = branch_error(y[~mask])
                errors[b_idx] = left_error + right_error
            # ------------------------------------------
            feature_breaks.append(breaks)
            feature_break_errors.append(errors)
    # ---------------------------------------------------------------------------------------------
    # Find the (feature, breakpoint) pair with the lowest total error
    best_error = np.inf
    split_feature = 0
    split_b_idx = 0
    for f in range(num_features):
        local_min_idx = np.argmin(feature_break_errors[f])
        local_min     = feature_break_errors[f][local_min_idx]
        if local_min < best_error:
            best_error    = local_min
            split_feature = f
            split_b_idx   = local_min_idx
        else : # if errors are all the same, all 0 then advisable to split across the defaults (0th feature, 0th breakpoint.)
            split_feature = rnd.randint(0, num_features-1) # This causes dependency on order in which features appear. Better to pick randomly?
            split_b_idx = len(feature_breaks[split_feature])//2 # split by middle value of 0th feature's breaks. Works even if 0th feature has only one breakpoint.

    split_value = feature_breaks[split_feature][split_b_idx]
    # Partition x and y on the best split
    mask        = x[split_feature] <= split_value
    left_x,  left_y  = x[:, mask],  y[mask]
    right_x, right_y = x[:, ~mask], y[~mask]
    return split_feature, split_value, left_x, left_y, right_x, right_y # some of these may be empty.

# -------------------------------------------------------------------------------------------------

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
        DT.insert(np.array([None, None, prediction]))
        return DT

    # Find the best split
    split_feature, split_value, left_x, left_y, right_x, right_y = make_error_array(x, y)
    print(f"sf, sv, lx, rx, = {split_feature}, {split_value}, {left_x}, {right_x}")
    if left_x.shape[1] == 0 or right_x.shape[1] == 0 : # should it just be "empty"?
        print(f"False split! Will make leaf.")
        prediction = mean_prediction(y)
        DT.insert(np.array([None, None, prediction]))
        return DT
    else :
        # Recurse into left and right subtrees
        left_subtree  = build_tree(left_x,  left_y,  smallest_class)
        right_subtree = build_tree(right_x, right_y, smallest_class)
        DT.insert(np.array([split_feature, split_value, None]))
        DT.root.left  = left_subtree.root
        DT.root.right = right_subtree.root
    return DT