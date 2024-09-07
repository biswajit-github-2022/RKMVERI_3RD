library(readr)
data <- read_csv("D:/x_MSC/3rd_sem/class/Econometry_and_Finance/Finance/Assignments/assignment_3/selected_2/combined_returns.csv")
View(data)


# Assume your data frame is named 'df'
# Check the shape of the data frame
dim(data)

# Calculate the variance-covariance matrix
cov_matrix <- cov(data)

# Print the variance-covariance matrix
print(cov_matrix)

# Check the dimensions of the variance-covariance matrix
dim(cov_matrix)

mean_vector <- colMeans(data)

# Print the mean vector
print(mean_vector)

# Check the dimensions of the mean vector
length(mean_vector)
#-------------------------------------------------------------------------------------
#Find Weights


# Install and load the quadprog package if you haven't already
install.packages("quadprog")
library(quadprog)

V <- cov_matrix 

# Set up Dmat, dvec, Amat, bvec
Dmat <- 2 * V
dvec <- rep(0, nrow(V))  # Coefficients for the linear term, here it's 0

# Equality constraint: sum(w) = 1
Amat <- cbind(rep(1, nrow(V)), diag(nrow(V)))

# Right-hand side for constraints
bvec <- c(1, rep(0, nrow(V)))

# Solve the quadratic programming problem
solution <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)

# Extract the solution
w <- solution$solution

# Print the result
print(w)
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------

# Unbounded closed form solution

# Assume you already have 'cov_matrix' (10x10) and 'mean_vector' (length 10)
# Create the 1_vector (length 10)
one_vector <- rep(1, length(mean_vector))

# Calculate the inverse of the covariance matrix
inv_cov_matrix <- solve(cov_matrix)

# Calculate A = mean_vector's Transpose * cov_matrix's Inverse * mean_vector
A <- t(mean_vector) %*% inv_cov_matrix %*% mean_vector

# Calculate B = mean_vector's Transpose * cov_matrix's Inverse * 1_vector
B <- t(mean_vector) %*% inv_cov_matrix %*% one_vector

# Calculate C = 1_vector's Transpose * cov_matrix's Inverse * 1_vector
C <- t(one_vector) %*% inv_cov_matrix %*% one_vector

# Calculate D = A * C - B^2
D <- A * C - B^2

# Print results
print("A:")
print(A)
print("B:")
print(B)
print("C:")
print(C)
print("D:")
print(D)

# Print dimensions of A, B, C, D
cat("Dimension of A:", dim(A), "\n")
cat("Dimension of B:", dim(B), "\n")
cat("Dimension of C:", dim(C), "\n")
cat("Dimension of D:", dim(D), "\n")

b_values <- seq(-2, 2, length.out = 100)
sigma_min_sq_values <- numeric(length(b_values))
# Calculate sigma_min_sq for each b_value in b_values
for (i in seq_along(b_values)) {
  sigma_min_sq_values[i] <- (C / D) * (b_values[i] - (B / C))^2 + (1 / C)
}

# Print the sequence of sigma_min_sq values
print(sigma_min_sq_values)

# Optionally, you can check the length and structure of sigma_min_sq_values
cat("Length of sigma_min_sq_values:", length(sigma_min_sq_values), "\n")
cat("Structure of sigma_min_sq_values:\n")
str(sigma_min_sq_values)

plot( sigma_min_sq_values, b_values, type = "l", col = "blue", ylim=c(-0.0015,0.003),
      xlab = "Minimum Variance", ylab = "Portfolio Return (b)",
      main = "Minimum Variance vs. Portfolio Return")

w_mvp <-(inv_cov_matrix %*% one_vector)/C[1,1]
w_mvp
#-----------------------------------------------------------------------------------
# Load the necessary library
library(quadprog)

# Assuming V (variance-covariance matrix) and mu (mean vector of returns) are predefined
# V <- <your covariance matrix>
# mu <- <your mean returns vector>

# Number of assets
mu<-mean_vector
V<-cov_matrix

n <- length(mu)

b_values <- seq(min(mu), max(mu), length.out = 50)

# Initialize a vector to store minimum variances
min_variances <- numeric(length(b_values))

# Loop through each b and solve the optimization problem
for (i in seq_along(b_values)) {
  b <- b_values[i]
  
  # Define the constraints
  Dmat <-  2*V
  dvec <- rep(0, nrow(V))
  Amat <- cbind(rep(1, n), diag(n), mu)
  bvec <- c(1,rep(0,10), b)
  
  # Solve the quadratic programming problem
  solution <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)
  
  # Store the minimum variance (objective function value)
  min_variances[i] <- solution$value
}

# Plot the minimum variances against the values of b
plot(min_variances,b_values,  type = "l", col = "blue",
     ylab = "Portfolio Return (b)", xlab = "Minimum Variance",
     main = "Minimum Variance vs. Portfolio Return")


min(min_variances)

x <- 1:length(data$Return1)  # X-axis values
y <- data$Return5
plot(x, y, type = "l", col = "blue", lwd = 2, xlab = "X-axis", ylab = "Y-axis", main = "Line Plot Example")
