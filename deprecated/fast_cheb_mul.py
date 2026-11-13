import numpy as np
from numpy import float128 as f128
from scipy.fft import next_fast_len,dct
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct
# type 3 is for evaluating chebyshev polynomials at convenient points (cosines of uniformly spaced angles theta)...
# kind of, it multiplies by 2 every non-constant coefficient
# inverse also isn't exactly inverse, hence the /m and /2 for a[0]

def fast_cheb_mul(_a,_b):
	a=np.array(_a,dtype=f128); b=np.array(_b,dtype=f128)
	z=len(a)+len(b)-1
	m=next_fast_len(z) # this is for faster fft, bc they depend on the length for the dnc
	a[1:]/=2; b[1:]/=2
	a=dct(a,3,m); b=dct(b,3,m)
	a*=b
	a=dct(a,2)
	a/=m
	a[0]/=2
	return a[0:z]