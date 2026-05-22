import numpy as np 



u = np.array([1.0, 2.0])
v = np.array([3.0, -1.0])
w = np.array([0.5, 4.0])
W = np.array([2.0, 3.0])


## $<x,y>w = \sum_{i,j} w_{i,j}x_{i,j}y_{i,j}$
def inner_w(x, y): 
    return np.sum(W * x * y);

print(inner_w(u,v) == inner_w(v,u)) 