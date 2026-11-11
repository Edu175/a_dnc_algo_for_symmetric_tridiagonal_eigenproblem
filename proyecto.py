import numpy as np
from givens import givens_rot

def diag(T):
    N,_=T.shape

def _ZEROS(N,M):
    return np.zeros((N,M),dtype=np.float128)

def rotate(L):#O(n)
    L=L[1:]+L[:1]

def defl(H,Q,P,tol): #O(n^2)
    N,_=H.shape
    eta=min(1/(100*N),tol)
    tau=6*eta*N**2
    delt=tau*np.linalg.norm(H,ord=2)
    ND=0
    P[1:N].sort(key=lambda i: H[i,i])
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

            G=np.array([[c,s],[-s,c]],dtype=np.float128)
            
            Q[:,[P[i],P[i+1]]]=Q[:,[P[i],P[i+1]]]@G.T

            rotate(P[i+1:N-ND])
            ND+=1
        else: i+=1
    return ND


def FMM_autovalores(H,P):
    return "sorry"

        
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

    D1,Q1,P1=_DnC_ST(T1,tol)
    D2,Q2,P2=_DnC_ST(T2,tol)
    l1=Q1[-1,:]*bk1
    f2=Q2[0,:]*bk2

    H=np.block([
                [a,            l1.T,            f2.T],
                [l1,             D1, _ZEROS(k,N-k-1)]
                [f2, _ZEROS(N-k-1,k),             D2]
               ])
    Q=np.block([
                [    _ZEROS(k,1),              Q1, _ZEROS(k,N-k-1)],
                [      np.eye(1),     _ZEROS(1,k), _ZEROS(1,N-k-1)],
                [_ZEROS(N-k-1,1), _ZEROS(N-k-1,k),              Q2]
                ])
    P=[0]+[1+i for i in P1]+[1+len(P1)+i for i in P2]
    
    ND=defl(H,Q,P,tol)

    ls=FMM_autovalores(H,P[:N-ND],tol)
    ds=[H[P[i],P[i]] for i in range(1,N-ND)]

    z=[PHI3(i,ls,ds) for i in range(1,N)]  #\sum O(n^2*N/n)=O(N^2)


    for i in range(N):
        qs=[Q[P[i],P[j]] for j in range(1,N-ND)]
        vs1=PHI1(ls,z,ds,qs)
        vs2=PHI2(ls,z,ds)
        for j in range(N-ND):
            Q[P[i],P[j]]=(-qs[0]+vs1[j])/np.sqrt(1+vs2[j])
    H[:N-ND,:N-ND]=np.diag(ls)

    return H,Q,P
    


def SimTridiag(T,tol):
    N,_=T.shape
    for i in range(1,N):
        if(abs(T[i][i-1]-T[i-1][i])<=tol):
            return False
        for j in range(i-1):
            if(abs(T[i][j])>tol):
                return False
    return True

def DnC_SimTridiag(_T, dfl_tol=np.float128(1e-9)):
    T=np.copy(_T)
    T.dtype=np.float128
    (N,M)=T.shape
    if(N!=M):
        print("Error: T no es cuadrada")
        return T
    if(not SimTridiag(T,dfl_tol)):
        print("Error: T no es simétrica tridiagonal")
        return T
    return _DnC_ST(_T,dfl_tol)
