from numpy import block,random,diagflat
from numpy import float128 as f128
from deflation import _ZEROS
from test_script import test_matrix, get_random_SimTridiag

def mk_SimTridiag(D,OD):
    T=diagflat(D).astype(f128)
    N,_=T.shape
    for i in range(N-1):
        T[i,i+1],T[i+1,i]=OD[i],OD[i]
    assert(T.dtype==f128)
    return T

def Wilkinson_matrix(N): #W+_N=[1,w_i,1], w_i=|(N+1)/2-i| para i=1 ... N
    w=[abs((N-1)/2-i) for i in range(N)]
    return mk_SimTridiag(w,[1]*(N-1))

def generalized_glued_Wilkinson_matrix(n,k,g): #Matriz tridiagonal (nk) x (nk), nxn a bloques, W+_{g,k,n}=[g, W+_k ,g]
    W=_ZEROS(n*k,n*k)
    for i in range(n):
        W[i*k:(i+1)*k,i*k:(i+1)*k]=Wilkinson_matrix(k)
    for i in range(n-1):
        W[(i+1)*k,(i+1)*k-1],W[(i+1)*k-1,(i+1)*k] = g,g
    return W

def glued_Wilkinson_matrix(N,g): #k=25
    assert(N%25==0)
    return generalized_glued_Wilkinson_matrix(N//25,25,g)

def Toeplitz_matrix(N): #[1,2,1]
    return mk_SimTridiag([2]*N,[1]*(N-1))

def gamma_matrix(N,g): #[1,i*g,1]
    gam=[i*g for i in range(1,N+1)]
    return mk_SimTridiag(gam,[1]*(N-1))

def gamma2_matrix(N,g):#[1/100,1+i*g,1/100]
    gam=[1+i*g for i in range(1,N+1)]
    return mk_SimTridiag(gam,[1/100]*(N-1))

for N in [128,256,512]:
    msg='random en [-1,1] '+str(N)+'x'+str(N)
    test_matrix(get_random_SimTridiag(N,-1,1),msg=msg)

for N in [129,257,513]:
    msg='Wilkinson '+str(N)+'x'+str(N)
    test_matrix(Wilkinson_matrix(N),msg=msg)

for N in [125,275,525]:
    msg='Wilkinson pegadas '+str(N)+'x'+str(N)
    test_matrix(glued_Wilkinson_matrix(N,f128(10)**-14),msg=msg)

for N in [128,256,512]:
    msg='Toeplitz '+str(N)+'x'+str(N)
    test_matrix(Toeplitz_matrix(N),msg=msg)

for N in [128,256,512]:
    msg='gamma '+str(N)+'x'+str(N)
    test_matrix(gamma_matrix(N,f128(10)**-6),msg=msg)

for N in [128,256,512]:
    msg='gamma2 '+str(N)+'x'+str(N)
    test_matrix(gamma2_matrix(N,f128(10)**-6),msg=msg)