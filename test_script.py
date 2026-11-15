import time
import numpy as np
from numpy import float128 as f128
from norms import vnorm2
from test_matrices import get_test_matrix_groups

def print_table_header():
    print(f"| {'Matrix':<28} | {'N':>4} | {'Time (s)':>9} | {'Residual':>14} | {'Orthogonality':>14} |")
    print(f"|{'-'*30}|{'-'*6}|{'-'*11}|{'-'*16}|{'-'*16}|")

def print_table_sep():
    print(f"|{'-'*30}|{'-'*6}|{'-'*11}|{'-'*16}|{'-'*16}|")

def measure_solver(A, solver_fn, res_scale_fn, orth_scale_fn, msg="", time_fmt="9.2f", ignorePrint=True):
    N, _ = A.shape
    t0 = time.time()
    L, X = solver_fn(A)
    dt = time.time() - t0

    if not ignorePrint:
        got = X @ L @ X.T
        print(A)
        print('XLX^T:')
        print(got)

    normA = f128(np.linalg.norm(A.astype(np.float64), 2))
    R = A @ X - X @ L if L.ndim == 2 else A @ X - X * L
    O = X.T @ X - np.eye(N, dtype=X.dtype)

    max_res_error = max(vnorm2(R[:, i]) for i in range(N)) / (normA if normA > 0 else f128(1))
    max_orth_error = max(vnorm2(O[:, i]) for i in range(N))

    max_res_error /= res_scale_fn(N)
    max_orth_error /= orth_scale_fn(N)

    print(f"| {msg:<28} | {N:>4} | {dt:{time_fmt}} | {float(max_res_error):>14.4e} | {float(max_orth_error):>14.4e} |", flush=True)

def run_benchmark(test_fn):
    print_table_header()
    for group in get_test_matrix_groups():
        for A, msg in group:
            test_fn(A, msg=msg)
        print_table_sep()
