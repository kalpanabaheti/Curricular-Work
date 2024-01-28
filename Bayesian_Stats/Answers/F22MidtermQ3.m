clear all
close all force
randn('state',4);
%
sigma2 = 1;
ysum=-0.63;
n=10;
%mu = 1;
%
theta = 0;
thetas =[theta];
tau2 = 1;
taus=[tau2];

lambda = 1;
lambdas=[lambda];
burn =1000;
ntotal = 100000 + burn;
tic
for i = 1: ntotal
  sig = sqrt(1/(n+tau2));
  mu=ysum/(n+(1/tau2));
  %normal parametrized in matlab by mean and standard deviation, not mean and variance
  theta = normrnd(mu,sig);
  
  %gamma is parametrized in matlab by shape and scale, not shape and rate 
  x=gamrnd(1.5,1./(0.5*theta^2+1));
  
  tau=1./x;
thetas =[thetas; theta];
taus = [taus; tau2];

end
toc
% Q3.1
hist(thetas(burn+1:end), 40)

% Q3.2
mean_theta=mean(thetas(burn+1:end))

% Q3.3

EQT_CS = [quantile(thetas(burn+1:end),0.025), quantile(thetas(burn+1:end),0.975)]

 