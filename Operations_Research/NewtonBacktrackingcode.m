clc
clear all
close all
x0=[-1.2;1];
epsilon=10^-4;
alphabar=1;
c=1*10^-5;
rho=0.9;

f=@(x1,x2) 100*(x2-x1^2)^2+(1-x1)^2;
gradf=@(x1,x2) [400*(x1^3-x1*x2)+2*(x1-1);200*(x2-x1^2)];
hessf=@(x1,x2) [1200*x1^2-400*x2+2 -400*x1; -400*x1 200];

d=-inv(hessf(x0(1),x0(2)))*gradf(x0(1),x0(2));

x=x0;
z=x+alphabar*d;
iter=0;
while norm (gradf(x(1),x(2)))>epsilon
    alpha=alphabar;
    while f(z(1),z(2))>f(x(1),x(2))+(c*alpha)*transpose(gradf(x(1),x(2)))*d
        alpha=rho*alpha;
        z=x+alpha*d;    
    end
    iter=iter+1
    alpha
    x=x+alpha*d
    
    
    d=-inv(hessf(x(1),x(2)))*gradf(x(1),x(2));
end

