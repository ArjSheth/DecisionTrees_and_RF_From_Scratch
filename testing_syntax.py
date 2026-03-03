import DecisionTree_draft as DT_arrays
import RandomForests as rf
import random as rnd
from numpy import array as arrayy
x = arrayy([[175,158, 188, 176, 182, 176, 166, 192], [65, 62, 102, 98, 87, 52, 56, 104], [5,2,2,4,4,4,6,7]])
y = arrayy([1,0,0,0,1,0,1,1])

mytrees = rf.train_random_forest(x,y,3,20, 1)


rfpredictor = rf.predictor_function(mytrees)

A = [(arrayy([x[0,i], x[1,i], x[2,i]]) ,y[i]) for i in range(len(y))]


# for tree in mytrees:
#     for i in A :
#         print(tree.predict(i[0]), i[1])
#     print()
#     print()
for i in A :
    print(rfpredictor(i[0]), i[1])

# ---------------------------------------------------------------------------------------------------------
print("--"*30)
# mytree = DT_arrays.build_tree(x,y,1)
#
# print()
# print("--"*30)
# for i in A :
#     print(mytree.predict(i[0]), i[1])
# ---------------------------------------------------------------------------------------------------------

