import numpy as np
from numpy import float64 as f128
from jacobi22 import jacobi22
def diag(T):
    n,_ = T.shape
    if n==1 : return T,np.eye(1,dtype=f128)
    c,s= jacobi22(T)
    Q=np.array([[c,-s],[s,c]],dtype=f128)
    return Q.T@T@Q, Q

def _ZEROS(N,M):
    return np.zeros((N,M),dtype=f128)

def rotate(L):#O(n)
    L=L[1:]+L[:1]

def defl(H,Q,tol): #O(n^2)
    N,_=H.shape
    P=[i for i in range(N)]
    P[1:N].sort(key=lambda i: H[i,i])
    eta=min(1/(100*N),tol)
    tau=6*eta*N**2
    delt=tau*np.linalg.norm(H,ord=2)
    ND=0
    i=1
    while i<N-ND :
        if abs(H[P[0],P[i]])<delt: 
            H[P[0],P[i]],H[P[0],P[i]]=0,0
            rotate(P[i:N-ND])
            ND+=1
        
        else: i+=1            
    i=1     
    while i<N-ND-1: 
        if abs(H[P[i],P[i]]-H[P[i+1],P[i+1]])<delt:
            zj=H[P[0],P[i]],zi=H[P[0],P[i+1]]
            r=np.sqrt(zj**2+zi**2)
            c=zj/r
            s=zi/r
            dj=H[P[i],P[i]]
            di=H[P[i+1],P[i+1]]

            H2=np.copy(H)
            H[P[i],P[i]]=dj*c**2+di*s**2
            H[P[i+1],P[i+1]]=dj*s**2+di*c**2
            
            H[P[0],P[i]],H[P[i],P[0]]=r,r
            H[P[0],P[i+1]],H[P[i+1],P[0]]=0,0
            
            assert(np.allclose(H,H2,rtol=tol,atol=tol))

            G=np.array([[c,s],[-s,c]],dtype=f128)
            
            Q[:,[P[i],P[i+1]]]=Q[:,[P[i],P[i+1]]]@G.T

            rotate(P[i+1:N-ND])
            ND+=1
        else: i+=1
    #H=[_H,  0]
    #  [0 , _L]
    X=np.copy(Q)
    _H=_ZEROS(N-ND,N-ND)
    _L=_ZEROS(ND,ND)
    for i in range(N-ND):
        _H[i,i]=H[P[i],P[i]]
        _H[0,i]=H[P[0],P[i]]
        _H[i,0]=_H[0,i]
    for i in range(N-ND,N):
        _L[i-(N-ND),i-(N-ND)]=H[P[i],P[i]]
    for i in range(N):
        for j in range(N): Q[i,j]=X[P[i],P[j]]

    return _H,_L

def FMM_autovalores(H,):
    rt,_ =np.linalg.eig(H)
    rt.sort()
    return rt

def PHI1(ls,z,ds,qs):
    n=ls.shape[0]
    rt=[f128(0)]*n
    ev =lambda i, j : z[j]*qs[j+1]/(ds[j]-ls[i])
    for i in range(0,n):
        L=f128(0)
        R=f128(0)
        for j in range(0,n-1):
            if ds[j]<ls[i] : L+=ev(i,j)
            else: R+=ev(i,j)
        rt[i]=L+R
    return rt

def PHI2(ls,z,ds):
    n=ls.shape[0]
    rt=[f128(0)]*n
    ev =lambda i, j : (z[j]/(ds[j]-ls[i]))**2
    for i in range(0,n):
        for j in range(0,n-1):
            rt[i]+=ev(i,j)
    return rt

def PHI3(i,ls,ds,zi): #O(n^2) 
    n=ls.shape[0]
    if(i>=n or i<0): return 0.
    rt=(ds[i]-ls[0])*(ls[-1]-ds[i])
    for j in range(0,i):
        rt*=(ds[i]-ls[j+1])/(ds[i]-ds[j])
    for j in range(i,n-2):
        rt*=(ls[j+1]-ds[i])/(ds[j+1]-ds[i])
    rt=np.sqrt(rt)
    return rt if zi>=0 else -rt

def _DnC_ST(T,tol):
    N,_=T.shape
    if(N<=2):
        return diag(T)

    k=N//2
    T1=T[:k,:k] # k x k
    a=T[k,k]
    bk1=T[k,k-1]
    bk2=T[k,k+1]
    T2=T[k+1:,k+1:] # N-k-1 x N-k-1

    D1,Q1=_DnC_ST(T1,tol)
    D2,Q2=_DnC_ST(T2,tol)
    l1=Q1[[-1],:]*bk1
    f2=Q2[[0],:]*bk2
    

    H=np.block([
                [a,            l1.T,            f2.T],
                [l1,             D1, _ZEROS(k,N-k-1)],
                [f2, _ZEROS(N-k-1,k),             D2]
               ])
    Q=np.block([
                [    _ZEROS(k,1),              Q1, _ZEROS(k,N-k-1)],
                [      np.eye(1),     _ZEROS(1,k), _ZEROS(1,N-k-1)],
                [_ZEROS(N-k-1,1), _ZEROS(N-k-1,k),              Q2]
                ])
    
    _H, _L=defl(H,Q,tol)
    N,_=_H.shape
    ND,_= _L.shape

    ls=FMM_autovalores(_H)
    ds=[_H[i,i] for i in range(1,N)]

    z=[PHI3(i,ls,ds,_H[0,i+1]) for i in range(N-1)]  #\sum O(n^2*N/n)=O(N^2)
    for i in range(N):
        qs=np.copy(Q[i,:])
        vs1=PHI1(ls,z,ds,qs)
        vs2=PHI2(ls,z,ds)
        for j in range(N):
            Q[i,j]=(-qs[0]+vs1[j])/np.sqrt(1+vs2[j])
    H=np.block([
        [np.diagflat(ls), _ZEROS(N,ND)],
        [_ZEROS(ND,N)   ,           _L]
    ])
    return H,Q
    

def SimTridiag(T,tol):
    N,_=T.shape
    for i in range(1,N):
        if(abs(T[i][i-1]-T[i-1][i])>tol):
            return False
        for j in range(i-1):
            if(abs(T[i][j])>tol):
                return False
    return True

def DnC_SimTridiag(_T, dfl_tol=f128(1e-9)):
    T=np.copy(_T)
    T.dtype=f128
    N,M=T.shape
    if(N!=M):
        print("Error: T no es cuadrada")
        return T
    if(not SimTridiag(T,dfl_tol)):
        print("Error: T no es simétrica tridiagonal")
        return T
    return _DnC_ST(_T,dfl_tol)


A=np.array([[2,-1,0],[-1,3,5],[0,5,1]],dtype=f128)

L,X=DnC_SimTridiag(A)

print(X@L@X.T)
print(A)
assert(np.allclose(A,X@L@X.T,rtol=f128(1e-9),atol=f128(1e-9)))
