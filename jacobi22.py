from numpy import sqrt
from numpy import float128 as f128

EPS=f128(10)**-10

def jacobi22(A):
    if(abs(A[0,1])<=EPS): return f128(1.),f128(0.)
    tau=(A[1,1]-A[0,0])/(2*A[1,0])
    if(tau>=0): t=-1/(tau+sqrt(tau**2+1))
    else: t=1/(-tau+sqrt(tau**2+1))
    c=1/sqrt(1+t**2)
    s=t*s
    return c,s