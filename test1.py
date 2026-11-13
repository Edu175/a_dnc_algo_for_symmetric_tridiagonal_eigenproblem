import numpy as np
from numpy import float128 as f128
from proyecto import DnC_SimTridiag
EPS=f128(1e-5)
def test_matrix(A):
	L,X=DnC_SimTridiag(A)
	print('Testing Matrix A:')
	print(A)
	print('XLX^T:')
	print(X@L@X.T)
	assert(np.allclose(A,X@L@X.T,rtol=EPS,atol=EPS))
	print("Estan cerca :D\n\n")

def get_random_SimTridiag(n):
	A=np.zeros((n,n),dtype=f128)
	for i in range(n): A[i,i]=np.random.uniform(low=-100,high=100)
	for i in range(1,n):
		A[i-1,i]=A[i,i-1]=np.random.uniform(low=-100,high=100)
	return A

test_matrix(np.array([[2,-1,0],[-1,3,5],[0,5,1]],dtype=f128)) #3x3


test_matrix(np.array([[2,-1,0,0],[-1,3,5,0],[0,5,1,67],[0,0,67,-101]],dtype=f128)) #4x4

test_matrix(np.array([
            [2,-1,0,0,0],
            [-1,3,5,0,0],
            [0,5,1,67,0],
            [0,0,67,-101,0.03],
            [0,0,0,0.03,0]
            ],dtype=f128)) #5x5

test_matrix(np.array([
            [2,-1,0,0,0,0],
            [-1,3,5,0,0,0],
            [0,5,1,67,0,0],
            [0,0,67,-101,0.03,0],
            [0,0,0,0.03,0,999],
            [0,0,0,0,999,-0.286]
            ],dtype=f128)) #6x6

test_matrix(get_random_SimTridiag(28))