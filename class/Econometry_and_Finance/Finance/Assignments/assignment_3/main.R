library(readr)
data <- read_csv("C:/Users/biswajit/Downloads/archive/selected/combined_returns.csv")
print(data)


# Assume your data frame is named 'df'
# Check the shape of the data frame
dim(data)

# Calculate the variance-covariance matrix
cov_matrix <- cov(data)

# Print the variance-covariance matrix
print(cov_matrix)

# Check the dimensions of the variance-covariance matrix
dim(cov_matrix)


#-------------------------------------------------------------------------------------
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
