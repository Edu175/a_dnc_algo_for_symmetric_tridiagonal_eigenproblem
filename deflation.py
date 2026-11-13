from numpy import array, copy, sqrt, zeros
from numpy import float128 as f128
from norms import norm2
def _ZEROS(N,M):
    return zeros((N,M),dtype=f128)

def rotate(L):#O(n)
    return L[1:]+L[:1]


def deflation(H,Q,tol): #O(n^2)
    N,_=H.shape
    P=[i for i in range(N)]
    P[1:]=sorted(P[1:],key=lambda i: H[i,i])
    
    eta=min(1/(100*N),tol)
    tau=6*eta*N**2
    delt=tau*norm2(H)
    ND=0
    i=1
    while i<N-ND :
        if abs(H[P[0],P[i]])<delt: 
            H[P[0],P[i]],H[P[0],P[i]]=0,0
            P[i:N-ND]=rotate(P[i:N-ND])
            ND+=1
        
        else: i+=1            
    i=1     
    while i<N-ND-1: 
        if abs(H[P[i],P[i]]-H[P[i+1],P[i+1]])<delt:
            zj=H[P[0],P[i]],zi=H[P[0],P[i+1]]
            r=sqrt(zj**2+zi**2)
            c=zj/r
            s=zi/r
            dj=H[P[i],P[i]]
            di=H[P[i+1],P[i+1]]

            H[P[i],P[i]]=dj*c**2+di*s**2
            H[P[i+1],P[i+1]]=dj*s**2+di*c**2
            
            H[P[0],P[i]],H[P[i],P[0]]=r,r
            H[P[0],P[i+1]],H[P[i+1],P[0]]=0,0
            
            G=array([[c,s],[-s,c]],dtype=f128)
            
            Q[:,[P[i],P[i+1]]]=Q[:,[P[i],P[i+1]]]@G.T

            P[i+1:N-ND]=rotate(P[i+1:N-ND])
            ND+=1
        else: i+=1
    #H=[_H,  0]
    #  [0 , _L]
    X=copy(Q)
    _H=_ZEROS(N-ND,N-ND)
    _L=_ZEROS(ND,ND)
    for i in range(N-ND):
        _H[i,i]=H[P[i],P[i]]
        _H[0,i]=H[P[0],P[i]]
        _H[i,0]=_H[0,i]
    for i in range(N-ND,N):
        _L[i-(N-ND),i-(N-ND)]=H[P[i],P[i]]
    for i in range(N):
        for j in range(N): Q[i,j]=X[i,P[j]]
   
    return _H,_L
