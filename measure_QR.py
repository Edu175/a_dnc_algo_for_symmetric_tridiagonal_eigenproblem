import numpy as np
from scipy.linalg import lapack
from test_script import measure_solver, run_benchmark

eps=np.finfo(np.float64).eps

def solve_qr_dsteqr(A):
    """
    Resuelve el problema de autovalores tridiagonal simétrico usando
    DSTEV / DSTEQR de LAPACK (el mismo algoritmo QR usado en [1]).
    """
    A64=np.asarray(A, dtype=np.float64)
    d=np.diag(A64).copy()
    e=np.diag(A64,1).copy()
    vals,vecs,info=lapack.dstev(d,e,compute_v=1)
    assert info==0,f"LAPACK dstev (DSTEQR) falló con info={info}"
    return vals,vecs

def test_matrix_qr(A, msg=""):
    A64 = np.asarray(A, dtype=np.float64)
    measure_solver(
        A64,
        solver_fn=solve_qr_dsteqr,
        res_scale_fn=lambda N: N*eps,
        orth_scale_fn=lambda N: N*eps,
        msg=msg,
        time_fmt="9.4f",
    )

if __name__ == "__main__":
    run_benchmark(test_matrix_qr)
