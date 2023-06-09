import numpy as np

# Task 3: Load data from CSV files
matrix_A = np.genfromtxt('Data/Task_3_matrix_A.csv', delimiter=',')
matrix_B = np.genfromtxt('Data/Task_3_matrix_B.csv', delimiter=',')

# Task 3: Create a similarity matrix based on Matrix A and Matrix B
def calculate_cosine_similarity(matrix_1, matrix_2):
    dot_product = np.dot(matrix_1, matrix_2.T)
    norm_product = np.linalg.norm(matrix_1, axis=1)[:, np.newaxis] * np.linalg.norm(matrix_2, axis=1)
    similarity = dot_product / norm_product
    return similarity

similarity_matrix = calculate_cosine_similarity(matrix_A, matrix_B)
print("Similarity Matrix:")
print(similarity_matrix)

# Task 3: Calculate pairwise Euclidean distance between rows of Matrix A
euclidean_distances = np.linalg.norm(matrix_A[:, np.newaxis, :] - matrix_A, axis=2)
print("\nPairwise Euclidean Distances:")
print(euclidean_distances)

# Task 3: Apply Singular Value Decomposition (SVD) to Matrix A and reconstruct it using reduced rank
U, S, V = np.linalg.svd(matrix_A)
reduced_rank = 2  # Specify the desired reduced rank for reconstruction
reconstructed_matrix_A = U[:, :reduced_rank] @ np.diag(S[:reduced_rank]) @ V[:reduced_rank, :]
print("\nReconstructed Matrix A (Reduced Rank", reduced_rank, "):")
print(reconstructed_matrix_A)

# Task 3: Calculate the Frobenius norm of Matrix A
frobenius_norm_A = np.linalg.norm(matrix_A)
print("\nFrobenius Norm of Matrix A:", frobenius_norm_A, "≈", round(frobenius_norm_A, 2))

# Task 3: Calculate the Frobenius norm of Matrix B
frobenius_norm_B = np.linalg.norm(matrix_B)
print("\nFrobenius Norm of Matrix B:", frobenius_norm_B, "≈", round(frobenius_norm_B, 2))
