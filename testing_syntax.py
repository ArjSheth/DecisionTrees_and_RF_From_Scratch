import DecisionTree_UsingArrays as DT_arrays
from numpy import array as arrayy
x = arrayy([[175,158, 188, 176, 182, 176, 166, 192], [65, 62, 102, 98, 87, 52, 56, 104]])
y = arrayy([1,0,0,0,1,0,1,1])
mytree = DT_arrays.build_tree(x,y,1)

A = [(arrayy([x[0,i], x[1,i]]) ,y[i]) for i in range(len(y))]

for i in A :
    print(mytree.predict(i[0]), i[1])