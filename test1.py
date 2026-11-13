import numpy as np
from numpy import float64 as f128
from proyecto import DnC_SimTridiag
EPS=f128(1e-9)
A=np.array([[2,-1,0],[-1,3,5],[0,5,1]],dtype=f128) #3x3

L,X=DnC_SimTridiag(A)

print(X@L@X.T)
print(A)
assert(np.allclose(A,X@L@X.T,rtol=EPS,atol=EPS))

A=np.array([[2,-1,0,0],[-1,3,5,0],[0,5,1,67],[0,0,67,-101]],dtype=f128) #4x4

L,X=DnC_SimTridiag(A)

print(X@L@X.T)
print(A)
assert(np.allclose(A,X@L@X.T,rtol=EPS,atol=EPS))

A=np.array([
            [2,-1,0,0,0],
            [-1,3,5,0,0],
            [0,5,1,67,0],
            [0,0,67,-101,0.03],
            [0,0,0,0.03,0]
            ],dtype=f128) #5x5

L,X=DnC_SimTridiag(A)

print(X@L@X.T)
print(A)
assert(np.allclose(A,X@L@X.T,rtol=EPS,atol=EPS))



A=np.array([
            [2,-1,0,0,0,0],
            [-1,3,5,0,0,0],
            [0,5,1,67,0,0],
            [0,0,67,-101,0.03,0],
            [0,0,0,0.03,0,999],
            [0,0,0,0,999,-0.286]
            ],dtype=f128) #6x6

L,X=DnC_SimTridiag(A)

print(X@L@X.T)
print(A)
assert(np.allclose(A,X@L@X.T,rtol=EPS,atol=EPS))