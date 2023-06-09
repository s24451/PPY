import numpy as np


matrix_A = np.genfromtxt('Data/Task_4A.csv', delimiter=' ')
matrix_B = np.genfromtxt('Data/Task_4B.csv', delimiter=',')
matrix_C = np.genfromtxt('Data/Task_4C.csv', delimiter=',')

# Normalization
mean_A = np.mean(matrix_A, axis=0)
std_A = np.std(matrix_A, axis=0)
normalized_matrix_A = (matrix_A - mean_A) / std_A

print("Normalized Matrix:")
print(np.round(normalized_matrix_A, 2))

# covariance matrix for A
try:
    covariance_matrix_A = np.cov(normalized_matrix_A, rowvar=False)
    print("\nCovariance Matrix (A):")
    print(np.round(covariance_matrix_A, 2))
except ZeroDivisionError:
    print("\nCannot calculate covariance matrix (A): Division by zero encountered.")




# multiply B and C with numpy
try:
    np_multiplication_result_BC = np.matmul(matrix_B, matrix_C)
    print("\nMatrix Multiplication Result (NumPy) for B and C:")
    print(np.round(np_multiplication_result_BC, 2))
except ValueError:
    print("\nMatrix Multiplication Result (NumPy) for B and C: Error encountered.")

# function to manually multiply
def matrix_multiply(A, B):
    n, m1 = A.shape
    m2, p = B.shape
    if m1 != m2:
        raise ValueError("Matrices are not aligned for multiplication")
    result = [[0 for _ in range(p)] for _ in range(n)]

    for i in range(n):
        for j in range(p):
            for k in range(m1):
                result[i][j] += A[i][k] * B[k][j]
    return result

manual_result=matrix_multiply(matrix_B, matrix_C)
numpy_result=np_multiplication_result_BC
try:
    comparison = np.allclose(manual_result, numpy_result, atol=1e-8)
    if comparison:
        print("\nThe results are the same for the manual and NumPy multiplication methods.")
    else:
        print("\nThe results are different for the manual and NumPy multiplication methods.")
except ValueError:
    print("\nCannot compare the results of the manual and NumPy multiplication methods.")


#logarithm for matrix A
logarithm_result_A = None 
try:
    with np.errstate(divide='raise'):
        logarithm_result_A = np.log(matrix_A)
        logarithm_result_A[~np.isfinite(logarithm_result_A)] = np.nan
except (ZeroDivisionError, FloatingPointError):
    print("\nCannot calculate logarithm (A): Division by zero encountered.")

if logarithm_result_A is not None and np.isnan(logarithm_result_A).any():
    print("\nMatrix Logarithm (A): Error encountered.")
elif logarithm_result_A is not None:
    print("\nMatrix Logarithm (A):")
    print(np.round(logarithm_result_A, 2))


#logarithm for matrix B
logarithm_result_B = None 
try:
    with np.errstate(divide='raise'):
        logarithm_result_B = np.log(matrix_B)
        logarithm_result_B[~np.isfinite(logarithm_result_B)] = np.nan
except (ZeroDivisionError, FloatingPointError):
    print("\nCannot calculate logarithm (B): Division by zero encountered.")

if np.isnan(logarithm_result_B).any():
    print("\nMatrix Logarithm (B): Error encountered.")
else:
    print("\nMatrix Logarithm (B):")
    print(np.round(logarithm_result_B, 2))

#logarithm for matrix C
logarithm_result_C = None  
try:
    with np.errstate(divide='raise'):
        logarithm_result_C = np.log(matrix_C)
        logarithm_result_C[~np.isfinite(logarithm_result_C)] = np.nan
except (ZeroDivisionError, FloatingPointError):
    print("\nCannot calculate logarithm (C): Division by zero encountered.")

if logarithm_result_C is not None and np.isnan(logarithm_result_C).any():
    print("\nMatrix Logarithm (C): Error encountered.")
elif logarithm_result_C is not None:
    print("\nMatrix Logarithm (C):")
    print(np.round(logarithm_result_C, 2))
