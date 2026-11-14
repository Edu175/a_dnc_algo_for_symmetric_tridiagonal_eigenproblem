import numpy as np
from numpy import float128 as f128
from proyecto import DnC_SimTridiag
from norms import vnorm2
EPS=f128(1e-5)

rng=np.random.default_rng(175)

def test_matrix(A,msg="",ignorePrint=False):
	print('Testing Matrix',msg)
	N,_=A.shape
	L,X=DnC_SimTridiag(A)
	if not ignorePrint:
		got=X@L@X.T
		print(A)
		print('XLX^T:')
		print(got)
	normA=f128(np.linalg.norm(A.astype(np.float64),2))
	R=A@X-X@L
	O=X.T@X-np.eye(N,dtype=f128)
	max_res_error=max(vnorm2(R[:,i]) for i in range(N))/(normA if normA>0 else f128(1))
	max_orth_error=max(vnorm2(O[:,i]) for i in range(N))
	print('Residual error (max_i ||Tx_i - l_i x_i||_2 / ||T||_2): ',max_res_error)
	print('Orthogonality error (max_i ||X^T x_i - e_i||_2): ',max_orth_error)
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