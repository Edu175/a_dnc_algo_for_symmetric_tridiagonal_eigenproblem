import numpy as np
from numpy import float128 as f128

def eval_suma(_terms): # dnc (or binary tree) sum for better precision
	return np.sum(np.array(_terms),dtype=f128) # it does the same
	if not(len(_terms)): return 0
	p=0
	terms=[i for i in _terms]
	while len(terms)-p>1:
		a=terms[p]; p+=1
		b=terms[p]; p+=1
		terms.append(a+b)
	return terms[p]

def eval_prod(_terms): # dnc (or binary tree) prod for better precision
	# return np.prod(np.array(_terms),dtype=f128) # im not sure if it does the same
	if not(len(_terms)): return 1
	p=0
	terms=[i for i in _terms]
	while len(terms)-p>1:
		a=terms[p]; p+=1
		b=terms[p]; p+=1
		terms.append(a*b)
	return terms[p]

def phi_bisec(mu, delt, z):
	n= z.shape[0]
	return eval_suma([(z[i]**2)/(delt[i]-mu) for i in range(n)])

def PHI1(ls,z,ds,qs):
	n=ls.shape[0]
	rt=[f128(0)]*n
	ev =lambda i, j : z[j]*qs[j+1]/(ds[j]-ls[i])
	for i in range(0,n):
		L=[]
		R=[]
		for j in range(0,n-1):
			if ds[j]<ls[i] : L.append(ev(i,j))
			else: R.append(ev(i,j))
		rt[i]=eval_suma(L)+eval_suma(R)
	return rt

def PHI2(ls,z,ds):
	n=ls.shape[0]
	rt=[f128(0)]*n
	ev =lambda i, j : (z[j]/(ds[j]-ls[i]))**2
	for i in range(0,n):
		rt[i]=eval_suma([ev(i,j) for j in range(n-1)])
	return rt

def PHI3(i,ls,ds,zi):
	n=ls.shape[0]
	if(i>=n or i<0): return 0.
	rt=(ds[i]-ls[0])*(ls[-1]-ds[i])
	rt*=eval_prod([(ds[i]-ls[j+1])/(ds[i]-ds[j]) for j in range(i)])
	rt*=eval_prod([(ls[j+1]-ds[i])/(ds[j+1]-ds[i]) for j in range(i,n-2)])
	rt=np.sqrt(rt)
	return rt if zi>=0 else -rt
