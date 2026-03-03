import DecisionTree_UsingArrays as DT_arrays
import RandomForests as rf
import random as rnd
from numpy import array as arrayy
x = arrayy([[175,158, 188, 176, 182, 176, 166, 192], [65, 62, 102, 98, 87, 52, 56, 104], [12,12,1,1,2,6,12,10]])
y = arrayy([1,0,0,0,1,0,1,1])

mytrees = rf.train_random_forest(x,y,1,100, 2)
print(mytrees)

rfpredictor = rf.predictor_function(mytrees)

A = [(arrayy([x[0,i], x[1,i], x[2,i]]) ,y[i]) for i in range(len(y))]


for tree in mytrees:
    for i in A :
        print(tree.predict(i[0]), i[1])
    print()
    print()
for i in A :
    print(rfpredictor(i[0]), i[1])

# ---------------------------------------------------------------------------------------------------------

# mytree = DT_arrays.build_tree(x,y,4)

# A = [(arrayy([x[0,i], x[1,i]]) ,y[i]) for i in range(len(y))]

# for i in A :
#     print(mytree.predict(i[0]), i[1])
# ---------------------------------------------------------------------------------------------------------
def sampler(num_of_features : int, syllabus_size : int, num_of_trees : int) :
    box = []
    for iterator in range(num_of_trees):
        box.append(rnd.sample(range(num_of_features), syllabus_size))
    return box

# print(sampler(100, 4, 3))

