from numpy import sqrt
from numpy import sign
from numpy import f128

EPS=f128(10)**-10

def givens_rot(a,b):
    if abs(b)<=EPS:
        return 1.,0.
    if abs(b)>abs(a):
        t=a/b
        s=-sign(b)/sqrt(1+t**2)
        c=s*t
    else:
        t=b/a
        c=sign(a)/sqrt(1+t**2)
        s=c*t
    return c,s