from numpy import sqrt,float64
def norm2(v):
    rt=float64(0)
    for xi in v:
        rt+=xi**2
    return sqrt(rt)