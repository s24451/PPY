import numpy as np


matrix_A = np.genfromtxt('Data/Task_3_matrix_A.csv', delimiter=',')
matrix_B = np.genfromtxt('Data/Task_3_matrix_B.csv', delimiter=',')

#similarity matrix
def calculate_cosine_similarity(matrix_1, matrix_2):
    dot_product = np.dot(matrix_1, matrix_2.T)
    norm_product = np.linalg.norm(matrix_1, axis=1)[:, np.newaxis] * np.linalg.norm(matrix_2, axis=1)
    similarity = dot_product / norm_product
    return similarity

similarity_matrix = calculate_cosine_similarity(matrix_A, matrix_B)
print("Similarity Matrix:")
print(similarity_matrix)

# Euclidean distance
euclidean_distances = np.linalg.norm(matrix_A[:, np.newaxis, :] - matrix_A, axis=2)
print("\nPairwise Euclidean Distances:")
print(euclidean_distances)

# SVD
U, S, V = np.linalg.svd(matrix_A)
reduced_rank = 2 
reconstructed_matrix_A = U[:, :reduced_rank] @ np.diag(S[:reduced_rank]) @ V[:reduced_rank, :]
print("\nReconstructed Matrix A (Reduced Rank", reduced_rank, "):")
print(reconstructed_matrix_A)

#Frobenius norm of Matrix A
frobenius_norm_A = np.linalg.norm(matrix_A)
print("\nFrobenius Norm of Matrix A:", frobenius_norm_A, "≈", round(frobenius_norm_A, 2))

#Frobenius norm of Matrix B
frobenius_norm_B = np.linalg.norm(matrix_B)
print("\nFrobenius Norm of Matrix B:", frobenius_norm_B, "≈", round(frobenius_norm_B, 2))
