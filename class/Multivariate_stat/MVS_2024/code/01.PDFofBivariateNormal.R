library("mvtnorm")
library("lattice")
#library("plot3D")
library("GA")
x1=seq(-2,2,.05)
x2=seq(-2,2,.05)
Rho = -0.4
Mu1=0; S1=1
Mu2=0; S2=1

Mu=rbind(Mu1,Mu2)
CovarianceMatrix=matrix(c(S1,sqrt(S1*S2)*Rho,sqrt(S1*S2)*Rho,S2),2)
Grid=expand.grid(x1, x2)                   
z=dmvnorm(as.matrix(Grid), Mu, CovarianceMatrix)  
f=matrix(z,length(x1))
persp3D(x1,x2,f,shade=.5,theta=30, phi=10,ticktype = "detailed",expand=.4,d=10)

Points=rmvnorm(1000,mean = Mu,sigma = CovarianceMatrix)
plot(Points[,1],Points[,2],xlab = "X1",ylab = "X2")

