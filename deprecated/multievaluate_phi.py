import numpy as np
from collections import deque
from numpy import float128 as f128
from fast_cheb_mul import fast_cheb_mul
import finufft
# multievaluates sum_{k=-n}^n c_k e^{i*k*theta}

def sum_poly_fractions(Ps,Qs): # sums fraction polynomials Ps[i]/Qs[i] in O(nlog^2n)
	Ps=deque(Ps)
	Qs=deque(Qs)
	while len(Ps) > 1:
		P1=Ps[0]; Ps.popleft()
		Q1=Qs[0]; Qs.popleft()
		
		P2=Ps[0]; Ps.popleft()
		Q2=Qs[0]; Qs.popleft()

		Pnew=fast_cheb_mul(P1,Q2)+fast_cheb_mul(P2,Q1)
		Qnew=fast_cheb_mul(Q1,Q2)
		Ps.append(Pnew)
		Qs.append(Qnew)
			
	assert len(Ps)==1 and len(Qs)==1
	return Ps[0],Qs[0]

def multievaluate_cheb_pol(_xs,P):
	# using cos(theta) = (e^{i*theta} + e^{-i*theta}) / 2
	# to then use finufft
	thetas=np.arccos(_xs)
	coefs=np.block([P[::-1],P[1:]]) # P_k ... P_1 P_0 P_1 ... P_k
	coefs=np.array([complex(i) for i in coefs])
	ret=finufft.nufft1d2(thetas,coefs,eps=1e-14)
	ret=(ret+P[0])/2
	return ret.real

def multievaluate_phi(a, b, xs, which):
	r"""
	Multievaluates
	
	if which==1: \sum_{k=1}^{n-1} \frac{a_k}{b_k-x}
	if which==2: \sum_{k=1}^{n-1} \frac{a_k}{(b_k-x)^2}
	
	at $n$ points $x_i$.
	
	In O(nlog^2n).
	"""
	# build the polynomials P(x)/Q(x)
	n=len(xs)
	assert len(a)==n-1 and len(b)==n-1
	Ps=np.array([ [a[k]] for k in range(n-1)])
	if which==1: Qs=np.array([ [b[k],-1] for k in range(n-1)])
	elif which==2: Qs=np.array([ [b[k]**2 +1/2, -2*b[k] , 1/2] for k in range(n-1)],dtype=f128)
	else: assert False
	P,Q=sum_poly_fractions(Ps,Qs)
	nums=multievaluate_cheb_pol(xs,P)
	dens=multievaluate_cheb_pol(xs,Q)
	
	ret=np.array([nums[i]/dens[i] for i in range(n)])
	return ret