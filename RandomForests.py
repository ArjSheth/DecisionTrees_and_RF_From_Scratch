import numpy as np
import DecisionTree_draft as dectree
import random as rnd

"""
I have decision trees. Now :
1. Given data, select features at random. In particular, make 'm' batches of size 'k' each
2. Train 'm' decision trees, one on each batch.
3. Get prediction.
"""

# CHECK WHY SORTING IS NEEDED IN RANDOM SAMPLES. DOES THE ORDER IN WHICH FEATURES ARE LISTED MAKE A BIG DIFFERENCE? WHY?
# THIS TIES BACK TO HOW NO `POSSIBLE SPLIT` IS HANDLED AT THE TREE LEVEL.


def sampler(num_of_features : int, syllabus_size : int, num_of_trees : int) :
    box = []
    for iterator in range(num_of_trees):
        box.append(rnd.sample(range(num_of_features), syllabus_size)) # This sorting is needed. WHY?????
    return box # This is a list of lists

def train_rf(x,y, syllabus_size, num_of_trees, smallest_class = 1) :
    n_features, n_samples = x.shape
    sampler_result = sampler(n_features, syllabus_size, num_of_trees)
    iterator = 0
    while iterator < num_of_trees :
        temp_x = np.zeros((syllabus_size, n_samples))
        temp_y = np.zeros((n_samples,))
        temp_y[:] = y[iterator]

    pass


def train_random_forest(x,y, syllabus_size, num_of_trees, smallest_class = 1)->list[dectree.Tree] :
    """
    x.shape = (features,samples)
    y.shape = (samples,1)
    """
    num_of_features = x.shape[0]
    print(f"num_of_features = {num_of_features}")
    feature_indices_for_each_tree = sampler(num_of_features, syllabus_size, num_of_trees)
    print(f"sampler gives \n{feature_indices_for_each_tree}\n\n")
    iterator = 0
    my_trees = []
    while iterator < num_of_trees :
        print(f"x is {x}")
        print(f"y is {y}")
        print(f"feature_indices_for_each_tree[iterator] = {feature_indices_for_each_tree[iterator]}")

        temp_x = np.array([x[i] for i in feature_indices_for_each_tree[iterator]])
        temp_y = y # adjustment will occur later, at the level of the number of samples to be used.
        print(f"temp_x shape is {temp_x.shape}")
        print()
        print(f"temp_y shape is {temp_y.shape}")
        print()
        temp_tree = dectree.build_tree(temp_x,temp_y, smallest_class)
        my_trees.append(temp_tree)
        iterator += 1
        print("Added a tree")
        print()
        temp_x = None
        temp_y = None
        temp_tree = None
        pass
    # Now we have a list of trees. A random forest, if you will
    return my_trees

def predictor_function(ll : list[dectree.Tree]) :
    def predictor(x):
        interim_list = [ll[i].predict(x) for i in range(len(ll))]
        return sum(interim_list)/len(interim_list)
    return predictor
