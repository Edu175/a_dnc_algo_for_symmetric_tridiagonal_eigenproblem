import ctypes
import ctypes.util
import numpy as np
from test_script import measure_solver, run_benchmark

eps = np.finfo(np.float64).eps
_lapack_lib = ctypes.CDLL(ctypes.util.find_library("lapack"))

def solve_lapack_dnc(A):
    """
    Resuelve el problema de autovalores tridiagonal simétrico usando
    DSTEVD / DSTEDC de LAPACK (Divide & Conquer estabilizado con Gu y Eisenstat [1]).
    """
    A64=np.asarray(A, dtype=np.float64)
    N,_=A64.shape
    d=np.diag(A64).copy()
    e=np.diag(A64, 1).copy()
    Z=np.zeros((N, N), dtype=np.float64, order="F")

    n_c=ctypes.c_int(N)
    ldz_c=ctypes.c_int(N)
    info_c=ctypes.c_int(0)
    lwork=1+4*N+N*N
    liwork=3+5*N
    work=np.zeros(lwork, dtype=np.float64)
    iwork=np.zeros(liwork, dtype=np.int32)
    lwork_c=ctypes.c_int(lwork)
    liwork_c=ctypes.c_int(liwork)

    _lapack_lib.dstevd_(
        ctypes.byref(ctypes.c_char(b"V")),
        ctypes.byref(n_c),
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(ldz_c),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(lwork_c),
        iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(liwork_c),
        ctypes.byref(info_c),
    )
    assert info_c.value == 0, f"LAPACK dstevd_ (DSTEDC) falló con info={info_c.value}"
    return d, Z

def test_matrix_ref_dnc(A, msg=""):
    A64 = np.asarray(A, dtype=np.float64)
    measure_solver(
        A64,
        solver_fn=solve_lapack_dnc,
        res_scale_fn=lambda N: N * eps,
        orth_scale_fn=lambda N: N * eps,
        msg=msg,
        time_fmt="9.4f",
    )

if __name__ == "__main__":
    run_benchmark(test_matrix_ref_dnc)
