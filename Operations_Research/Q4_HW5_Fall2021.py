import cvxpy as cvx
import numpy as np

#inputs provides
p_min=np.array([20,20,10])
p_max=np.array([200,150,150])
f_max=np.array([100,110,50,80,60,40])
B=[11.6,5.9,13.7,9.8,5.6,10.5]
c=np.array([16,20,8])
d=[110,65,95]

#declaring variables
p=cvx.Variable(3)
f=cvx.Variable(6)
o=cvx.Variable(6)

#defining objective
objective = cvx.Minimize(p*c)

#defining constraints
constraints=  [f[0]-f[-1]==p[0],
               f[2]-f[1]==p[1],
               f[4]-f[3]==p[2],
               f[1]-f[0]==-d[0],
               f[3]-f[2]==-d[1],
               f[5]-f[4]==-d[2],
               
               f[0]==B[0]*(o[0]-o[1]),
               f[1]==B[1]*(o[1]-o[2]),
               f[2]==B[2]*(o[2]-o[3]),
               f[3]==B[3]*(o[3]-o[4]),
               f[4]==B[4]*(o[4]-o[5]),
               f[5]==B[5]*(o[5]-o[0]),
               
              -f_max<=f,
              f<=f_max,
              
              p_min<=p,
              p<=p_max,
              
            ]

myprob = cvx.Problem(objective, constraints)
myprob.solve()

#printing outputs
print("\nThe optimal value is", round(myprob.value,2))
print("The quantum of electricity produced at 1,3,5 are",
      [round(x,2) for x in p.value])
print("The flows in different lines are", 
[round(x,2) for x in f.value])
print("The electricty price at node 2 is", 
round(constraints[3].dual_value,2) )
print("The electricty price at node 4 is", 
round(constraints[4].dual_value,2) )
print("The electricty price at node 6 is", 
round(constraints[5].dual_value,2) )