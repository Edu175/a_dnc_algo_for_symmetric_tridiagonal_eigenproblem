from numpy import array
from numpy.linalg import eig
from numpy import float128 as f128
from norms import norm2
from phis import phi_bisec
def np_autovalores(H):
	rt,_ =eig(H)
	rt.sort()
	return rt

def bisec_autovalores(H,tol,ds):
    n,_= H.shape
    assert n==_
    if n==1: return array([H[0,0]])
    z=H[0,1:]
    a=H[0,0]
    eta=min(1/(101*n),tol) # en [1], sec 3.3 pag 180., se asume eta*n<.01
    ls=array([f128(0)]*n)
    delt=[f128(0)]*(n-1)
    n2z=norm2(z)
    for i in range(0,n):
        caso=0
        if i==0 or i==n-1:
            caso=(i!=0)
        else:

            mid=(ds[i-1]+ds[i])/2
            fmid=mid-a+phi_bisec(mid,ds[:i],z[:i])+phi_bisec(mid,ds[i:],z[i:])
            caso=(fmid>0)
        
        for j in range(n-1):
            delt[j]=ds[j]-ds[i-(caso)]
        ai=ds[i-(caso)]-a
        L=f128(0)
        R=f128(0)
        if(caso): 
            if(i<n-1): R=delt[i]/2
            else: R=n2z+abs(ai)
        else: 
            if i>0: L=delt[i-1]/2
            else: L=-n2z-abs(ai)

        while True: #va a terminar por el 'stop criterion' de [1], ec (9) sec 3.2, pag 178.
            mu=(L+R)/2
            psi=phi_bisec(mu,delt[:i],z[:i]) #para reducir error numérico, junto sumas del mismo signo
            phi=phi_bisec(mu,delt[i:],z[i:])
            gi=mu+ai+psi+phi

            if abs(gi)<=eta*n*(abs(mu)+abs(ai)+abs(psi)+abs(phi)): 
                ls[i]=mu+ds[i-caso]
                break
            elif gi<0: L=mu
            else: R=mu        
    return ls
