import numpy as np

# Load the CSV file
data = np.genfromtxt('Data\Task_1.csv', delimiter=';', skip_header=1)


X = data[:, :-1]
Y = data[:, -1]
num_cells = data.size
num_cells = data.size
num_rows, num_columns = data.shape
mean = np.mean(data)
median = np.median(data)
variance = np.var(data)
data_without_missing = data[~np.isnan(data)]
mean_without_missing = np.mean(data_without_missing)
median_without_missing = np.median(data_without_missing)
variance_without_missing = np.var(data_without_missing)
print("Number of cells:", num_cells)
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)
print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Mean without missing values:", mean_without_missing, "≈", round(mean_without_missing, 2))
print("Median without missing values:", median_without_missing, "≈", round(median_without_missing, 2))
print("Variance without missing values:", variance_without_missing, "≈", round(variance_without_missing, 2))

