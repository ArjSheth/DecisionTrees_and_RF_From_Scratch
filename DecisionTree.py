import numpy as np
# 1. get a tree
# 2. implement a function that takes training data and makes a tree based on it.
"""
Currently, need to add Leaf node concept. Also need to "BuildTree"

"""



class Node:
    def __init__(self, value):
        self.value = value
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
            raise "Node already exists, can only insert at 'root'"

    # def _insert_recursive(self, value, node):
    #     if value < node.value:
    #         if node.left is None:
    #             node.left = Node(value)
    #         else:
    #             self._insert_recursive(value, node.left)
    #     else:
    #         if node.right is None:
    #             node.right = Node(value)
    #         else:
    #             self._insert_recursive(value, node.right)
# -------------------------------------------------------------------------------------------------

def mean_prediction(y):
    # mean = 0
    # for value in y:
    #     mean += value
    # mean /= len(y)
    return np.mean(y)

def branch_error(y):
    error = 0
    if len(y) == 0:
        return error
    else:
        mean = mean_prediction(y)
        for value in y :
            error += (value - mean)**2
        return error

def make_error_array(x: np.array(), y: np.array()) :
    # x contains input data, each row a feature, y contains outputs.
    # assume x contains x[0], x[1], .., x[p-1], where x[0] is a record of observed values for the 0'th feature.
    # We must go over all features.

    num_of_features, num_of_samples = np.shape(x)

    feature_break_error_array = array()
    for feature in x:
        feature_index = x.index(feature)  # to be used for record-keeping
        # Sort feature
        sorted_feature = sorted(feature)
        breaks = [0.5*(sorted_feature[i]+sorted_feature[i+1]) for i in range(len(sorted_feature)-2)]
        for breakpoint in breaks :
            # Now the greedy algo.
            # Check all points in x for whom feature value is less than breakpoint.
            breakpoint_index = sorted_feature.index(breakpoint) # to be used for record-keeping
            left_y = []
            right_y = []
            for i in range(num_of_features):
                if x[feature[i]] <= breakpoint:
                    left_y.append(y[i])
                else :
                    right_y.append(y[i])
            left_error = branch_error(left_y)
            right_error = branch_error(right_y)
            branch_net_error = left_error + right_error
            feature_break_error_array[feature_index, breakpoint_index] = branch_net_error
        pass
    i = 0
    j = 0
    location_of_least_error = [i,j]
    least_error = feature_break_error_array[i, j]
    while i < len(feature_break_error_array):
        while j < len(feature_break_error_array):
            if feature_break_error_array[i, j] >= least_error:
                j += 1
            else :
                least_error = feature_break_error_array[i, j]
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
        if x[split_feature[i]] <= split_value:
            best_left_x.append(x[:,i])
            best_left_y.append(y[i])
        else:
            right_x.append(x[:, i])
            right_y.append(y[i])

    # Now we have found where the least error occurs FIRST.
    # Can toggle the equality in the above if condition to make it the LAST time.
    return location_of_least_error # this contains split feature and split value.
    # To use this, one must have x,y, location_of_least_error.
    # With these, one can recover left_branch and right_branch, and apply the same algo to them.

def build_tree(x:list[list[float]], y:list[float], location_of_least_error:list, smallest_class = 10):
    DT = Tree()
    if len(y) == 0:
        return DT
    elif len(x[0])*len(x) <= smallest_class:
        return DT # CHECK THIS, MIGHT NEED A LINE OR TWO
    else:
        pass