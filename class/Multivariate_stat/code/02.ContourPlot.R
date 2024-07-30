library("mvtnorm")
library("lattice")
#library("plot3D")
library("car")
x1=seq(-2,2,.05)
x2=seq(-2,2,.05)
Rho = 0.4
Mu1=0; S1=1
Mu2=0; S2=1

sample_size=1000
Mu=rbind(Mu1,Mu2)
CovarianceMatrix=matrix(c(S1,sqrt(S1*S2)*Rho,sqrt(S1*S2)*Rho,S2),2)
Grid=expand.grid(x1, x2)     
#Generating multivariate normal
z=rmvnorm(sample_size, Mu, CovarianceMatrix)  
#Scatter plot of bivariate normal
plot(z[,1],z[,2],main="Scatter Plot", sub="Bivariate Data",
     xlab="X1", ylab="X2",xlim=c(-4, 4), ylim=c(-4,4))
Sm=colMeans(z)
Svcm=var(z)
  
CovarianceMatrixInverse=solve(CovarianceMatrix) #For population version
SvcmInverse=solve(Svcm)                         #For sample version
# #Calculating Statistical Distance
# dist_vector=diag((z-rep(1,sample_size)%*%(t(Mu)))
#                  %*%CovarianceMatrixInverse%*%
#                    t(z-rep(1,sample_size)%*%(t(Mu))))

#Estimating Statistical Distance
dist_vector=diag((z-rep(1,sample_size)%*%(t(Sm)))
                 %*%SvcmInverse%*%
                   t(z-rep(1,sample_size)%*%(t(Sm)))) 

alpha=.05
#Calculating 100(1-alpha) percentile of chi-square(p) dist
chisq_value=qchisq(alpha,length(Mu), ncp = 0, lower.tail = FALSE)
#Contour plot of constant statistical distance (at median and at (1-alpha)%)
dataEllipse(z[,1], z[,2], levels=c(0.5, 1-alpha))
#Checking proportion of points inside 100(1-alpha)% contour
point_proportion=length(dist_vector[dist_vector<chisq_value])/length(dist_vector)
print(point_proportion)

