import numpy as np
# 1. get a tree
# 2. implement a function that takes training data and makes a tree based on it.
"""
Need to fix lines 99 onwards.
"""



class Node:
    def __init__(self, value):
        self.value = value # For our purpose, this will be [split_feature, split_value, left_prediction, right_prediction]
        self.left = None
        self.right = None

# Will mimic the typical tree class for decision trees.

class Tree:
    def __init__(self):
        self.root = None  # Starts empty

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            raise ValueError("Node already exists, can only insert at 'root'")
    def predict(self, x):
        def traverse_nodes(active_node, inpu) : # This way I don't need to treat each subtree as a tree.
            if active_node.value[0] is None :
                return 0.5*(active_node.value[2] + active_node.value[3]) # If not splitting, then we reached the critical size.
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

def mean_prediction(y): # will have y as a 1D array
    return np.mean(y)

def branch_error(y):
    if len(y) == 0:
        return 0
    mean = mean_prediction(y)
    return np.sum((y-mean)**2)


def make_error_array(x: np.array(), y: np.array()) :
    # x contains input data, each row a feature, y contains outputs.
    # assume x contains x[0], x[1], .., x[p-1], where x[0] is a record of observed values for the 0'th feature.
    # We must go over all features.

    num_of_features, num_of_samples = np.shape(x)
    feature_break_error_array = np.array([[float('inf') for _ in range(num_of_samples)] for _ in range(num_of_features)])
    for feature_index in range(num_of_features):
        feature = x[feature_index]
        # to be used for record-keeping
        # Sort feature
        sorted_feature_values = np.sort(feature)
        breaks = [0.5*(sorted_feature_values[i]+sorted_feature_values[i+1]) for i in range(len(sorted_feature_values)-2)]
        for breakpoint_index in range(len(breaks)) :
            breakpoint = breaks[breakpoint_index]
            # Now the greedy algo.
            # Check all points in x for whom feature value is less than breakpoint.
            # to be used for record-keeping
            left_y = []
            right_y = []
            for i in range(num_of_samples):
                if x[feature_index,i] <= breakpoint:
                    left_y.append(y[i])
                else :
                    right_y.append(y[i])
            left_error = branch_error(left_y)
            right_error = branch_error(right_y)
            branch_net_error = left_error + right_error
            feature_break_error_array[feature_index][breakpoint_index] = branch_net_error
        pass
    i = 0
    j = 0
    location_of_least_error = [i,j]
    least_error = feature_break_error_array[i][j]
    while i < len(feature_break_error_array):
        while j < len(feature_break_error_array[i]):
            if feature_break_error_array[i][j] >= least_error:
                j += 1
            else :
                least_error = feature_break_error_array[i][j]
                location_of_least_error = [i, j]
                j += 1
        i += 1
        j = 0
    best_left_x = np.array()
    best_right_x = np.array()
    best_left_y = np.array()
    best_right_y = np.array()
    split_feature = location_of_least_error[0]
    split_value = location_of_least_error[1]
    for i in range(num_of_samples):
        if x[split_feature,i] <= split_value:
            best_left_x += x[:,i]
            best_left_y += y[i]
        else:
            best_right_x += x[:, i]
            best_right_y += y[i]
    split_feature = location_of_least_error[0]
    split_value = location_of_least_error[1]
    # Now we have found where the least error occurs FIRST.
    # Can toggle the equality in the above if condition to make it the LAST time.
    return split_feature, split_value, best_left_x, best_left_y, best_right_x, best_right_y
    # With these, one can recover left_branch and right_branch, and apply the same algo to them.

def build_tree(x, y, smallest_class = 10):
    DT = Tree()

    if len(y) == 0:
        return DT

    # Stopping condition: too few samples → leaf node
    if x.shape[1] <= smallest_class:
        prediction = mean_prediction(y)
        # Leaf: split fields unused; both predictions hold the same constant estimate
        DT.insert(np.array([None, None, prediction, prediction]))
        return DT

    # Find the best split
    split_feature, split_value, left_x, left_y, right_x, right_y = make_error_array(x, y)

    # Internal node: prediction fields unused at this level
    DT.insert(np.array([split_feature, split_value, None, None], dtype=object))

    # Recurse into left and right subtrees
    left_subtree = build_tree(left_x, left_y, smallest_class)
    right_subtree = build_tree(right_x, right_y, smallest_class)

    DT.root.left = left_subtree.root
    DT.root.right = right_subtree.root

    return DT