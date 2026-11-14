import time
import numpy as np
from numpy import float128 as f128
from proyecto import DnC_SimTridiag
from norms import vnorm2

rng=np.random.default_rng(175)
dfl_tol=f128(1e-10)
eps=np.finfo(f128).eps

def print_table_header():
	print(f"| {'Matrix':<28} | {'N':>4} | {'Time (s)':>9} | {'Residual':>14} | {'Orthogonality':>14} |")
	print(f"|{'-'*30}|{'-'*6}|{'-'*11}|{'-'*16}|{'-'*16}|")

def print_table_sep():
	print(f"|{'-'*30}|{'-'*6}|{'-'*11}|{'-'*16}|{'-'*16}|")

def test_matrix(A,msg="",ignorePrint=False):
	N,_=A.shape
	t0=time.time()
	L,X=DnC_SimTridiag(A,dfl_tol)
	dt=time.time()-t0
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

	tau=6*dfl_tol*N*(np.log2(N)+1)
	max_res_error/=tau
	max_orth_error/=N*eps
	
	print(f"| {msg:<28} | {N:>4} | {dt:>9.2f} | {float(max_res_error):>14.4e} | {float(max_orth_error):>14.4e} |", flush=True)

def get_random_SimTridiag(n,Low=-100, High=100):
	A=np.zeros((n,n),dtype=f128)
	for i in range(n): A[i,i]=rng.uniform(low=Low,high=High)
	for i in range(1,n):
		A[i-1,i]=A[i,i-1]=rng.uniform(low=Low,high=High)
	return A

def test_random(n,ignorePrint=False):
	test_matrix(get_random_SimTridiag(n),str(n)+"x"+str(n),ignorePrint) # nxn