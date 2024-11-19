import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from cvxopt import matrix, solvers

# Load the consolidated returns CSV
df = pd.read_csv('consolidated_returns.csv')

# Input array of column indices (e.g., 2, 3, 5, 6, 10)
input_columns = [6, 8, 10, 14, 15]

# Select relevant columns (ignoring the 'game' column)
data_matrix = df.iloc[:, input_columns].to_numpy()
# data_matrix=0

print(data_matrix)
# 1. Dimensions of the data matrix
print("Dimensions of data matrix:", data_matrix.shape)

# 2. Variance-covariance matrix
cov_matrix = np.cov(data_matrix, rowvar=False)
print("Variance-Covariance Matrix:\n", cov_matrix)
print("Dimensions of Covariance Matrix:", cov_matrix.shape)

# 3. Mean vector
mean_vector = np.mean(data_matrix, axis=0)
print("Mean Vector:\n", mean_vector)
print("Length of Mean Vector:", len(mean_vector))

# 4. Unbounded closed-form solution
one_vector = np.ones(len(mean_vector))

# Inverse of covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Define constraints
num_assets = cov_matrix.shape[0]

# Dmat: Quadratic term (2 * covariance matrix)
Dmat = 2 * cov_matrix

# dvec: Linear term (set to 0 as there is no linear term in the objective function)
dvec = np.zeros(num_assets)

# Equality constraint: sum(weights) = 1
A = np.ones((1, num_assets))  # 1 x num_assets matrix for equality constraint
b = np.array([1.0])  # RHS for equality: sum(weights) = 1

# Inequality constraint: weights >= 0
G = -np.eye(num_assets)  # Negative identity matrix for weights >= 0
h = np.zeros(num_assets)  # RHS for weights >= 0

# Convert inputs to cvxopt format
Dmat = matrix(Dmat)
dvec = matrix(dvec)
G = matrix(G)
h = matrix(h)
A = matrix(A)
b = matrix(b)

# Solve quadratic programming problem
solution = solvers.qp(Dmat, dvec, G, h, A, b)  # A and b are equality constraints

# Extract the solution
weights = np.array(solution['x']).flatten()

# Print the optimal weights
print("Optimal Weights:", weights)