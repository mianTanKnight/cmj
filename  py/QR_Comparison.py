import numpy as np

eps = 1e-7  ## 10^{-7}

## A \in R^{4 \times 3}
A = np.array([[1.0, 1.0, 1.0],
              [eps, 0.0, 0.0],
              [0.0, eps, 0.0],
              [0.0, 0.0, eps]])

cond_A = np.linalg.cond(A)
print(f"==================================================")
print(f"矩阵 A 的条件数 K_2(A) = {cond_A:.2e}")
print(f"==================================================\n")


### CGS
def cgs(A: np.ndarray):
    m, n = A.shape
    # v_{q} (\in R^{m \times 1}) <- QR <- v_{a} (\in R^{n \times 1} )
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
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
    # v_{q} (\in R^{m \times 1}) <- QR <- v_{a} (\in R^{n \times 1} )
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        w = A[:, j].copy()
        for i in range(j):
            # update aj to w
            R[i, j] = np.dot(Q[:, i], w)
            w -=  R[i, j]* Q[:,i]
        R[j, j] = np.linalg.norm(w)
        Q[:, j] = w / R[j, j]
    return Q, R


# ----------------------------------------------------
# 运行对比测试
# ----------------------------------------------------
Q_cgs, R_cgs = cgs(A)
Q_mgs, R_mgs = mgs(A)
Q_np, R_np   = np.linalg.qr(A) # 基于 Householder 反射

# 计算正交性误差 ||I - Q^T * Q||_F (如果 Q 完美正交，结果应该接近 0)
I = np.eye(3)
err_cgs = np.linalg.norm(I - Q_cgs.T @ Q_cgs, 'fro')
err_mgs = np.linalg.norm(I - Q_mgs.T @ Q_mgs, 'fro')
err_np  = np.linalg.norm(I - Q_np.T @ Q_np, 'fro')

print(f"算法正交性误差对比 || I - Q^T * Q ||_F：")
print(f"1. CGS (经典格拉姆-施密特) 正交性误差 : {err_cgs:.2e}")
print(f"2. MGS (修改格拉姆-施密特) 正交性误差 : {err_mgs:.2e}")
print(f"3. NumPy/Householder 正交性误差       : {err_np:.2e}")