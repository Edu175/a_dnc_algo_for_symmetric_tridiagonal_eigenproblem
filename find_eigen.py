from numpy import array
from numpy.linalg import eig
from numpy import float128 as f128
from norms import norm2

def np_autovalores(H):
	rt,_ =eig(H)
	rt.sort()
	return rt

def phi_bisec(mu, delt, z):
	n= z.shape[0]
	rt=f128(0)
	for i in range(n):
		rt+=(z[i]**2)/(delt[i]-mu)
	return rt


def bisec_autovalores(H,tol,ds):
	n,_= H.shape
	z=H[0,1:]
	a=H[0,0]
	eta=min(1/(101*n),tol)
	ls=array([f128(0)]*n)
	for i in range(0,n):
		caso=0
		if i==0 or i==n-1:
			caso=(i!=0)
		else:

			mid=(ds[i-1]+ds[i])/2
			fmid=mid-a+phi_bisec(mid,ds[:i],z[:i])+phi_bisec(mid,ds[i:],z[i:])
			caso=(fmid>0)
		
		delt=[f128(0)]*(n-1)
		for j in range(n-1):
			delt[j]=ds[j]-ds[i-(caso)]
		ai=ds[i-(caso)]-a
		L=f128(0)
		R=f128(0)
		if(caso): 
			if(i<n-1): R=delt[i]/2
			else: R=norm2(z)+abs(i)
		else: 
			if i>0: L=delt[i-1]/2
			else: L=-norm2(z)-abs(ai)
		paso=0
		print('i =',i)
		print(n-1)
		while True:
			paso+=1
			mu=(L+R)/2
			psi=phi_bisec(mu,delt[:i],z[:i])
			phi=phi_bisec(mu,delt[i:],z[i:])
			gi=mu+ai+psi+phi
			lhs=abs(gi)
			rhs=eta*n*(abs(mu)+abs(ai)+abs(phi)+abs(phi))
			if paso %10000==0:
				print('paso',paso,'lhs',lhs,'rhs',rhs)
			if abs(gi)<=eta*n*(abs(mu)+abs(ai)+abs(phi)+abs(phi)): 
				ls[i]=mu+ds[i-caso]
				break
			elif gi<0: L=mu
			else: R=mu        
	return ls
