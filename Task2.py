import numpy as np

# Task 2: Upload data from "Task_2.csv"
data = np.genfromtxt('Data\Task_2.csv', delimiter=';')

# Task 2: Find eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(data)
print("Eigenvalues:")
print(np.round(eigenvalues, 2))
print("\nEigenvectors:")
print(np.round(eigenvectors, 2))

# Task 2: Calculate the inverse of the matrix
inverse_matrix = np.linalg.inv(data)
print("\nInverse Matrix:")
print(np.round(inverse_matrix, 2))

# Task 2: Implement a custom function to calculate the determinant
def custom_determinant(matrix):
    return np.round(np.linalg.det(matrix), 2)

determinant = custom_determinant(data)
print("\nDeterminant:", determinant)

# Task 2: Solve a linear system of equations
# Assuming 'b' is a column vector of right-hand side values
b = np.array([[10], [5], [12], [8], [9], [16], [7], [11], [14], [5]])
solution = np.linalg.solve(data, b)
print("\nLinear System Solution:")
print(np.round(solution, 2))

# Task 2: Compute the matrix rank
rank = np.linalg.matrix_rank(data)
print("\nMatrix Rank:", rank)
