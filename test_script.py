import numpy as np
from numpy import float128 as f128
from proyecto import DnC_SimTridiag
EPS=f128(1e-5)

rng=np.random.default_rng(175)

def test_matrix(A,msg="",ignorePrint=False):
	print('Testing Matrix',msg)
	L,X=DnC_SimTridiag(A)
	got=X@L@X.T
	if not ignorePrint:
		print(A)
		print('XLX^T:')
		print(got)
	max_abs_error=f128(0)
	max_rel_error=f128(0)
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			if A[i][j]>1e-15:
				abs_err=abs(A[i][j]-got[i][j])
				rel_err=abs_err/abs(A[i][j])
				max_abs_error=max(max_abs_error,abs_err)
				max_rel_error=max(max_rel_error,rel_err)
	print('Max absoulte error: ',max_abs_error)
	print('Max relative error: ',max_rel_error)
	# assert(np.allclose(A,got,rtol=EPS,atol=EPS))
	# print("Estan cerca :D")
	print("\n")

def get_random_SimTridiag(n,Low=-100, High=100):
	A=np.zeros((n,n),dtype=f128)
	for i in range(n): A[i,i]=rng.uniform(low=Low,high=High)
	for i in range(1,n):
		A[i-1,i]=A[i,i-1]=rng.uniform(low=Low,high=High)
	return A

# test_matrix(np.array([[2,-1,0],[-1,3,5],[0,5,1]],dtype=f128),msg="3x3") #3x3


# test_matrix(np.array([[2,-1,0,0],[-1,3,5,0],[0,5,1,67],[0,0,67,-101]],dtype=f128),"4x4") #4x4

'''
test_matrix(np.array([
            [2,-1,0,0,0],
            [-1,3,5,0,0],
            [0,5,1,67,0],
            [0,0,67,-101,0.03],
            [0,0,0,0.03,0]
            ],dtype=f128),"5x5") #5x5
#'''

'''
test_matrix(np.array([
            [2,-1,0,0,0,0],
            [-1,3,5,0,0,0],
            [0,5,1,67,0,0],
            [0,0,67,-101,0.03,0],
            [0,0,0,0.03,0,999],
            [0,0,0,0,999,-0.286]
            ],dtype=f128),"6x6") #6x6
#'''

def test_random(n,ignorePrint=False):
	test_matrix(get_random_SimTridiag(n),str(n)+"x"+str(n),ignorePrint) # nxn
	
test_random(1,ignorePrint=True)
test_random(1,ignorePrint=True)
test_random(28,ignorePrint=True)
test_random(40,ignorePrint=True)
test_random(50,ignorePrint=True)
test_random(75,ignorePrint=True)
test_random(90,ignorePrint=True)
test_random(100,ignorePrint=True)