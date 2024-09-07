# Install quadprog package
install.packages("quadprog")

# Load the quadprog library
library(quadprog)

#Minimize:  (1/2) x^T D x - d^T x 
#Subject to: A^T x >= b


# Define the matrix D (must be positive definite)
D <- matrix(c(2, 0, 0, 2), nrow = 2)

# Define the vector d
d <- c(1, 1)

# Define the matrix A (for inequality constraints)
A <- matrix(c(1, 0, 0, 1, -1, -1), nrow = 2)

# Define the vector b (right-hand side of constraints)
b <- c(1, 1, -2)

# Solve the QP problem
solution <- solve.QP(D, dvec = -d, Amat = A, bvec = b, meq = 0)

# Print the solution
print(solution$solution)


# Define the plot range
x <- seq(-2, 3, length.out = 100)
y <- seq(-2, 3, length.out = 100)

# Create a grid for plotting
z <- outer(x, y, function(x, y) 0.5 * (2 * x^2 + 2 * y^2) - x - y)

# Plot the contours of the objective function
contour(x, y, z, levels = pretty(z, 10), xlab = "x1", ylab = "x2", main = "Quadratic Programming Solution")

# Add inequality constraint lines
abline(h = 1, col = "blue", lty = 2)  # Constraint: x2 >= 1
abline(v = 1, col = "green", lty = 2)  # Constraint: x1 >= 1
abline(a = 2, b = -1, col = "red", lty = 2)  # Constraint: x1 + x2 >= 2

# Mark the solution point
points(solution$solution[1], solution$solution[2], col = "red", pch = 19, cex = 1.5)

# Add a legend
legend("topright", legend = c("Objective Function", "x2 >= 1", "x1 >= 1", "x1 + x2 >= 2", "Solution"),
       col = c("black", "blue", "green", "red", "red"), lty = c(1, 2, 2, 2, 0), pch = c(NA, NA, NA, NA, 19))

#------------------------------------------------------------------------------------------
