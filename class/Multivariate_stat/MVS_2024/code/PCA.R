library(readr)
data <- read_csv("D:/x_MSC/3rd_sem/class/Multivariate_stat/MVS_2024/code/national_track_records_for_men[1].csv")
View(data)


# Store the first column in a separate variable
first_column <- data[, 1]

# Store the rest of the columns in another variable
rest_of_data <- data[, -1]

# Calculate the mean vector for the remaining columns
mean_vector_rest <- colMeans(rest_of_data, na.rm = TRUE)

# Print the results
cat("First column:\n")
print(first_column)

cat("Mean vector of the rest of the data:\n")
print(mean_vector_rest)


# Calculate the correlation matrix for the remaining columns (rest_of_data)
correlation_matrix <- cor(rest_of_data, use = "complete.obs")  # 'use = "complete.obs"' handles missing values by omitting rows with NAs
#---------------
# Print the correlation matrix
cat("Correlation matrix of the rest of the data:\n")
print(correlation_matrix)


# Perform eigenvalue decomposition of the correlation matrix
eigen_decomposition <- eigen(correlation_matrix)

# Extract eigenvalues
eigen_values <- eigen_decomposition$values

# Extract eigenvectors
eigen_vectors <- eigen_decomposition$vectors

# Print the results
cat("Eigenvalues:\n")
print(eigen_values)

cat("Eigenvectors:\n")
print(eigen_vectors)

#--------------------
# Create a scree plot of the eigenvalues
plot(eigen_values, 
     type = "b",                   # Line and points
     xlab = "Principal Component",  # X-axis label
     ylab = "Eigenvalue",           # Y-axis label
     main = "Scree Plot",           # Title
     pch = 19,                      # Solid circle points
     col = "blue")                  # Color of points and lines

# Add gridlines for better readability (optional)
grid()
