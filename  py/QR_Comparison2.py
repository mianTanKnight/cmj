import numpy as np
from tabulate import tabulate

### CGS
def cgs(A: np.ndarray):
    m, n = A.shape
    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    # step1: a_{j}
    for j in range(n):
        w = A[:, j].copy()
        # R_{ij} = a_{j} - \sum_{i = 1}^{j - 1} R_{ij}q_{i}
        for i in range(j):
            R[i, j] = np.dot(Q[:,i],A[:, j])
            w -= R[i, j]* Q[:,i] # 减去 proj_{q_{i}}(a_{j})
        R[j, j] = np.linalg.norm(w)
        Q[:, j] = w/R[j,j]
    return Q, R

### MGS
def mgs(A: np.ndarray):
    m, n = A.shape
    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    for j in range(n):
        w = A[:, j].copy()
        for i in range(j):
            # update aj to w
            R[i, j] = np.dot(Q[:, i], w)
            w -=  R[i, j]* Q[:,i]
        R[j, j] = np.linalg.norm(w)
        Q[:, j] = w / R[j, j]
    return Q, R


def toffloat32():
    # ε ∈ {10^-1,10^-2,10^-3,10^-4,10^-5,10^-6,10^-8,10^-10,10^-12}
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
    eps_f32 = np.array(eps_list, dtype = np.float32)
    size = eps_f32.size
    records = [] 

    for k in range(size):
        current_eps = eps_f32[k]
        A = np.array([[1.0, 1.0, 1.0],
              [current_eps, 0.0, 0.0],
              [0.0,current_eps, 0.0],
              [0.0, 0.0, current_eps]], dtype = np.float32)
        cond_A = np.linalg.cond(A)

        Q_cgs, R_cgs = cgs(A)
        Q_mgs, R_mgs = mgs(A)
        Q_np, R_np   = np.linalg.qr(A)
        e_fact_cgs = np.float32(np.linalg.norm(A - Q_cgs @ R_cgs, ord="fro")) / np.float32(np.linalg.norm(A))

        e_fact_mgs = np.float32(np.linalg.norm(A - Q_mgs @ R_mgs, ord="fro")) / np.float32(np.linalg.norm(A))
        e_fact_np = np.float32(np.linalg.norm(A - Q_np @ R_np, ord="fro")) / np.float32(np.linalg.norm(A))

        I = np.eye(3, dtype=A.dtype)
        err_cgs = np.float32(np.linalg.norm(I - Q_cgs.T @ Q_cgs, 'fro'))
        err_mgs = np.float32(np.linalg.norm(I - Q_mgs.T @ Q_mgs, 'fro'))
        err_np  = np.float32(np.linalg.norm(I - Q_np.T @ Q_np, 'fro'))
        records.append([current_eps,cond_A,e_fact_cgs,e_fact_mgs,e_fact_np,err_cgs,err_mgs,err_np])

    headers = ["eps_32", "cond_A",  "Efact-CGS", "Efact-MGS", "Efact-NpQR", "Eorth-CGS", "Eorth-MGS", "Eorth-NpQR"]
    print(tabulate(records, headers=headers, floatfmt=".2e"))

def toffloat64():
    # ε ∈ {10^-1,10^-2,10^-3,10^-4,10^-5,10^-6,10^-8,10^-10,10^-12}
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
    eps_f64 = np.array(eps_list, dtype = np.float64)
    size = eps_f64.size
    records = [] 

    for k in range(size):
        current_eps = eps_f64[k]
        ## A \in R^{4 \times 3}
        A = np.array([[1.0, 1.0, 1.0],
              [current_eps, 0.0, 0.0],
              [0.0,current_eps, 0.0],
              [0.0, 0.0, current_eps]], dtype = np.float64)
        cond_A = np.linalg.cond(A)
        Q_cgs, R_cgs = cgs(A)
        Q_mgs, R_mgs = mgs(A)
        Q_np, R_np   = np.linalg.qr(A)
        e_fact_cgs = np.linalg.norm(A - Q_cgs @ R_cgs, ord="fro") / np.linalg.norm(A)
        e_fact_mgs = np.linalg.norm(A - Q_mgs @ R_mgs, ord="fro") / np.linalg.norm(A)
        e_fact_np = np.linalg.norm(A - Q_np @ R_np, ord="fro") / np.linalg.norm(A)

        I = np.eye(3, dtype=A.dtype)
        err_cgs = np.linalg.norm(I - Q_cgs.T @ Q_cgs, 'fro')
        err_mgs = np.linalg.norm(I - Q_mgs.T @ Q_mgs, 'fro')
        err_np  = np.linalg.norm(I - Q_np.T @ Q_np, 'fro')
        records.append([current_eps,cond_A,e_fact_cgs,e_fact_mgs,e_fact_np,err_cgs,err_mgs,err_np])

    headers = ["eps_64", "cond_A", "Efact-CGS", "Efact-MGS", "Efact-NpQR", "Eorth-CGS", "Eorth-MGS", "Eorth-NpQR"]
    print(tabulate(records, headers=headers, floatfmt=".2e"))

def main():
    toffloat32()
    toffloat64()
    pass 

# output ->
"""
  eps_32    cond_A    Efact-CGS    Efact-MGS    Efact-NpQR    Eorth-CGS    Eorth-MGS    Eorth-NpQR
--------  --------  -----------  -----------  ------------  -----------  -----------  ------------
1.00e-01  1.73e+01     2.23e-09     1.15e-09      1.69e-09     5.74e-06     8.16e-07      8.14e-09
1.00e-02  1.73e+02     4.49e-10     3.11e-10      3.44e-08     8.17e-04     1.16e-05      1.19e-07
1.00e-03  1.73e+03     1.72e-11     1.61e-11      5.80e-11     3.78e-02     5.35e-05      6.42e-08
1.00e-04  1.73e+04     3.37e-22     7.25e-15      7.06e-12     7.07e-01     1.15e-04      1.46e-08
1.00e-05  1.73e+05     1.95e-22     8.77e-14      2.34e-13     7.07e-01     1.15e-05      6.15e-08
1.00e-06  1.73e+06     6.31e-23     1.08e-15      2.52e-14     7.07e-01     1.16e-06      6.15e-08
1.00e-08  1.73e+08     1.77e-25     4.34e-17      1.45e-16     7.07e-01     1.05e-07      6.15e-08
1.00e-10  1.73e+10     1.05e-27     1.33e-18      4.29e-18     7.07e-01     6.76e-08      6.15e-08
1.00e-12  1.73e+12     1.38e-29     1.42e-20      6.98e-20     7.07e-01     8.63e-08      6.15e-08
  eps_64    cond_A    Efact-CGS    Efact-MGS    Efact-NpQR    Eorth-CGS    Eorth-MGS    Eorth-NpQR
--------  --------  -----------  -----------  ------------  -----------  -----------  ------------
1.00e-01  1.73e+01     2.83e-18     3.37e-18      5.56e-16     1.79e-14     2.51e-15      7.47e-16
1.00e-02  1.73e+02     2.67e-19     1.95e-19      6.41e-17     2.58e-12     3.65e-14      6.66e-16
1.00e-03  1.73e+03     2.44e-20     2.39e-20      3.85e-16     4.07e-11     5.76e-14      4.46e-16
1.00e-04  1.73e+04     4.30e-21     4.68e-21      1.93e-20     3.20e-09     4.53e-13      3.16e-16
1.00e-05  1.73e+05     3.57e-22     3.70e-22      1.93e-21     6.76e-08     9.57e-13      2.33e-16
1.00e-06  1.73e+06     1.98e-23     2.59e-23      6.41e-17     7.26e-05     1.03e-10      4.25e-16
1.00e-08  1.73e+08     8.34e-26     1.43e-25      3.21e-24     7.07e-01     1.15e-08      2.69e-16
1.00e-10  1.73e+10     2.96e-27     2.95e-27      1.71e-26     7.07e-01     1.15e-10      3.43e-16
1.00e-12  1.73e+12     1.51e-29     2.39e-29      3.59e-28     7.07e-01     1.15e-12      8.07e-16

"""
"""
我们分析下输出数据: 
efact 从 10^{-1} - 10^{-12}
我们发现 CGS, MGS, NpQR 的效果几乎相等 
但区别在于 32 和 64  64的效果要比32 强 9 - 12 个数量级

但在 eorth
CGS 的崩溃点在 10^{-4}(32), 10^{-8}(64) , MGS 和 NpQR 并没有崩溃
64的效果要比32 强 9 - 12 个数量级

"""


if __name__ == "__main__":
    main()
