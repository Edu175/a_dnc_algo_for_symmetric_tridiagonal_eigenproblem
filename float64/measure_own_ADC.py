import numpy as np
from numpy import float64 as f128
from proyecto import DnC_SimTridiag
from test_script import measure_solver, run_benchmark

dfl_tol=f128(1e-10)
eps=np.finfo(f128).eps

def test_matrix_own_adc(A, msg=""):
    measure_solver(
        A,
        solver_fn=lambda M: DnC_SimTridiag(M,dfl_tol),
        res_scale_fn=lambda N: 6*dfl_tol*N*(np.log2(N)+1),
        orth_scale_fn=lambda N: N * eps,
        msg=msg,
        time_fmt="9.2f",
    )

if __name__ == "__main__":
    run_benchmark(test_matrix_own_adc)
