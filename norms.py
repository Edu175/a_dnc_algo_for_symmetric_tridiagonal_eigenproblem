from numpy import sqrt, random, inner, dot, inf, zeros
from numpy import float128 as f128

# from scipy.sparse.linalg import svds
# spectral_norm = svds(A, k=1, return_singular_vectors=False)[0]

EPS=f128(1e-9)

def vnorm2(v):
	rt=f128(0)
	for xi in v:
		rt+=xi**2
	return sqrt(rt)

def mnorm2(A, isArrow=False):
	n,m=A.shape
	
	v=random.uniform(f128(-1), f128(1), size=n).astype(f128)
	v/=norm2(v)

	def mul(A,v):
		if not isArrow: return dot(A,v)
		else:
			ret=zeros(n,dtype=f128)
			ret[0]=inner(A[0,:],v)
			for i in range(1,n):
				ret[i]=A[i,0]*v[0] + A[i,i]*v[i]
			return ret
	ant=inf
	while True:
		u=mul(A.T,mul(A,v)) 
		lam=inner(v,u)
		if(abs(lam-ant)<=EPS): return lam
		v=u/norm2(u)
		ant=lam
		

def norm2(x,isArrow=False):
	if(x.ndim==1):
		assert not isArrow
		return vnorm2(x)
	return mnorm2(x)
	

