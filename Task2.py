import numpy as np


data = np.genfromtxt('Data\Task_2.csv', delimiter=';')

#eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(data)
print("Eigenvalues:")
print(np.round(eigenvalues, 2))
print("\nEigenvectors:")
print(np.round(eigenvectors, 2))

#the inverse of the matrix
inverse_matrix = np.linalg.inv(data)
print("\nInverse Matrix:")
print(np.round(inverse_matrix, 2))

#the determinant
def custom_determinant(matrix):
    return np.round(np.linalg.det(matrix), 2)

determinant = custom_determinant(data)
print("\nDeterminant:", determinant)

#linear system of equations
b = np.array([[10], [5], [12], [8], [9], [16], [7], [11], [14], [5]])
solution = np.linalg.solve(data, b)
print("\nLinear System Solution:")
print(np.round(solution, 2))

#the matrix rank
rank = np.linalg.matrix_rank(data)
print("\nMatrix Rank:", rank)
