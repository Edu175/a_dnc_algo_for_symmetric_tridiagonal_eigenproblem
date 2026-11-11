import numpy as np
from collections import deque
from numpy import float128 as f128
from scipy.signal import convolve

def sum_poly_fractions(Ps,Qs): # sums fraction polynomials Ps[i]/Qs[i] in O(nlog^2n)
	Ps=deque(Ps)
	Qs=deque(Qs)
	while len(Ps) > 1:
		P1=Ps[0]; Ps.popleft()
		Q1=Qs[0]; Qs.popleft()
		
		P2=Ps[0]; Ps.popleft()
		Q2=Qs[0]; Qs.popleft()

		Pnew=convolve(P1,Q2)+convolve(P2,Q1)
		Qnew=convolve(Q1,Q2)
		Ps.append(Pnew)
		Qs.append(Qnew)
			
	assert len(Ps)==1 and len(Qs)==1
	return Ps[0],Qs[0]

def multievaluate_phi(a, b, xs, which):
	r"""
	Multievaluates
	
	if which==1: \sum_{k=1}^{n-1} \frac{a_k}{b_k-x}
	if which==2: \sum_{k=1}^{n-1} \frac{a_k}{(b_k-x)^2}
	
	at $n$ points $x_i$.
	"""
	# build the polynomials P(x)/Q(x)
	n=len(xs)
	assert len(a)==n-1 and len(b)==n-1
	Ps=np.array([ [a[k]] for k in range(n-1)])
	Qs=np.array()
	if which==1: Qs=np.array([ [b[k],-1] for k in range(n-1)])
	elif which==2: Qs=np.array([ [b[k]**2, -2*b[k] , 1] for k in 
	range(n-1)])
	else: assert False
	P,Q=sum_poly_fractions(Ps,Qs)
	P=np.polynomial(P)
	Q=np.polynomial(Q)
	
	# multi-evaluate polynomials P and Q at the n points x_i
	# for now it's n^2, we could change this to the nlog^2n one
	nums=P(xs)
	dens=Q(xs)
	
	ret=np.array([nums[i]/dens[i] for i in range(n)])
	return ret