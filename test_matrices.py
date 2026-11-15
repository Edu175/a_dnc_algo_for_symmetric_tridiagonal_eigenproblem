import numpy as np
from numpy import diagflat
from numpy import float128 as f128
from deflation import _ZEROS

def get_random_SimTridiag(n, Low=-100, High=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(175)
    A = np.zeros((n, n), dtype=f128)
    for i in range(n):
        A[i, i] = rng.uniform(low=Low, high=High)
    for i in range(1, n):
        A[i-1, i] = A[i, i-1] = rng.uniform(low=Low, high=High)
    return A

def mk_SimTridiag(D, OD):
    T = diagflat(D).astype(f128)
    N, _ = T.shape
    for i in range(N-1):
        T[i, i+1], T[i+1, i] = OD[i], OD[i]
    assert T.dtype == f128
    return T

def Wilkinson_matrix(N): # W+_N=[1,w_i,1], w_i=|(N+1)/2-i| para i=1 ... N
    w = [abs((N-1)/2 - i) for i in range(N)]
    return mk_SimTridiag(w, [1]*(N-1))

def generalized_glued_Wilkinson_matrix(n, k, g): # Matriz tridiagonal (nk) x (nk), nxn a bloques, W+_{g,k,n}=[g, W+_k ,g]
    W = _ZEROS(n*k, n*k)
    for i in range(n):
        W[i*k:(i+1)*k, i*k:(i+1)*k] = Wilkinson_matrix(k)
    for i in range(n-1):
        W[(i+1)*k, (i+1)*k-1], W[(i+1)*k-1, (i+1)*k] = g, g
    return W

def glued_Wilkinson_matrix(N, g): # k=25
    assert N % 25 == 0
    return generalized_glued_Wilkinson_matrix(N // 25, 25, g)

def Toeplitz_matrix(N): # [1,2,1]
    return mk_SimTridiag([2]*N, [1]*(N-1))

def gamma_matrix(N, g): # [1,i*g,1]
    gam = [i*g for i in range(1, N+1)]
    return mk_SimTridiag(gam, [1]*(N-1))

def gamma2_matrix(N, g): # [1/100,1+i*g,1/100]
    gam = [1 + i*g for i in range(1, N+1)]
    return mk_SimTridiag(gam, [1/100]*(N-1))

def get_test_matrix_groups():
    rng = np.random.default_rng(175)
    return [
        [(get_random_SimTridiag(N, -1, 1, rng=rng), f'random en [-1,1] {N}x{N}') for N in [128, 256, 512]],
        [(Wilkinson_matrix(N), f'Wilkinson {N}x{N}') for N in [129, 257, 513]],
        [(glued_Wilkinson_matrix(N, f128(10)**-14), f'Wilkinson pegadas {N}x{N}') for N in [125, 275, 525]],
        [(Toeplitz_matrix(N), f'Toeplitz {N}x{N}') for N in [128, 256, 512]],
        [(gamma_matrix(N, f128(10)**-6), f'gamma {N}x{N}') for N in [128, 256, 512]],
        [(gamma2_matrix(N, f128(10)**-6), f'gamma2 {N}x{N}') for N in [128, 256, 512]],
    ]
