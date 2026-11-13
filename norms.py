from numpy import sqrt, random, inner, dot, nan
from numpy import float128 as f128


# from scipy.sparse.linalg import svds
# spectral_norm = svds(A, k=1, return_singular_vectors=False)[0]

EPS=f128(1e-9)

def vnorm2(v):
    rt=f128(0)
    for xi in v:
        rt+=xi**2
    return sqrt(rt)

def mnorm2(A):
    n,m=A.shape
    
    v=random.uniform(f128(-1), f128(1), size=n).astype(f128)
    v/=norm2(v)

    ant=nan #siempre tira falso, tito-truco
    while True:
        u=dot(A.T,dot(A,v)) 
        lam=inner(v,u)
        if(abs(lam-ant)<=EPS): return lam
        v=u/norm2(u)
        ant=lam
        

def norm2(x):
    if(x.ndim==1): return vnorm2(x)
    return mnorm2(x)
    
