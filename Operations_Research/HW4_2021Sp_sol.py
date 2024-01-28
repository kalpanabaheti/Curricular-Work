# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:47:39 2021

@author: kobre
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm
from pylab import meshgrid
from mpl_toolkits.mplot3d import Axes3D


# Question 1. Convex optimization

# Q1.1

# objective function
def func(x):
    return np.exp(x) - x

# construct lines
x = np.linspace(-2, 2, 2000)
# function values
yfunc = func(x)
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot()
# plot function
plt.plot(x, yfunc, label = r'$f(x) = e^x - 1$', zorder = 1, color = 'red')
plt.plot(0, 1, zorder = 3, color = 'red', marker = 'o')
# set axis limits
plt.xlim((-2, 2))
#plt.ylim((0, 6))
ax.grid(True, which = 'both')
ax.annotate(r'min = $(0, 1)$', (-0.3, 0.8), color = 'red', fontsize = 12)
# uncoment to save image
#plt.savefig('function.png')
plt.show()

# first order derivative of the function
def firstder(x):
    return np.exp(x) - 1

# second order derivative of the function
def secondder(x):
    return np.exp(x)

#Newthon's method
# stopping criteria  
err = 10**(-5)
# starting point
x_it = -1
it = 0
# first derivative at starting point
ffdx = firstder(x_it)
while np.abs(ffdx) >= err:
    #function value at current point
    fx = func(x_it)
    #value of the first derivative at current point
    ffdx = firstder(x_it)
    #value of the second derivative at current point
    fsdx = secondder(x_it)
    print("Iteration", it)
    print("Current point is", x_it)
    print("Function value at current point", fx)
    print("First derivative at current point", ffdx)
    print("Second derivative at current point", fsdx)
    x_it = x_it - ffdx/fsdx
    it = it + 1
    

# Q1.2
    
# construct lines
x = np.linspace(-1.5, 1.5, 2000)
# first derivative y = e^x - 1
yfuncder = np.exp(x) - 1
# tangent at x0
a0 = -1
b0 = firstder(a0)
k0 = secondder(a0)
y0 = k0*(x - a0) + b0
# print value x0
print("Point x0", a0)
# equation of a tangent at x0
print("Tangent at x0: y = ", k0, "*x + ", b0-k0*a0)
# tangent at x1
a1 = a0 - b0/k0
b1 = firstder(a1)
k1 = secondder(a1)
y1 = k1*(x - a1) + b1
# print value x1
print("Point x1", a1)
# equation of a tangent at x1
print("Tangent at x1: y = ", k1, "*x + ", b1-k1*a1)
# tangent at x2
a2 = a1 - b1/k1
b2 = firstder(a2)
k2 = secondder(a2)
y2 = k2*(x - a2) + b2
# print value x2
print("Point x2", a2)
# equation of a tangent at x2
print("Tangent at x2: y = ", k2, "*x + ", b2-k2*a2)

fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot()
ax.grid(True, which = 'both')
# plot curve, tangents, and points
plt.plot(x, yfuncder, label = r'$f(x) = e^x - 1$', zorder = 1, color = 'blue')
plt.plot(x, y0, label = r'tangent at $x^0$', zorder = 2, color = 'red', linestyle ='dashed')
plt.plot(a0, 0, zorder = 3, color = 'red', marker = 'o')
plt.plot(a0, b0, zorder = 4, color = 'red', marker = 'o')
plt.plot(x, y1, label = r'tangent at $x^1$', zorder = 5, color = 'green', linestyle = 'dashed')
plt.plot(a1, 0, zorder = 6, color = 'green', marker = 'o')
plt.plot(a1, b1, zorder = 7, color = 'green', marker = 'o')
plt.plot(x, y2, label = r'tangent at $x^2$', zorder = 8, color = 'purple', linestyle = 'dashed')
plt.plot(a2, 0, zorder = 9, color = 'purple', marker = 'o')
plt.plot(a2, b2, zorder = 10, color = 'purple', marker = 'o')
plt.plot(0, 0, zorder = 11, color = 'blue', marker = 'o')
# add legend
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0.)
# set limits for axis
plt.xlim((-1.5, 1.5))
plt.ylim((-1, 4))
# label points on the graph
ax.annotate(r'$a^0$', (-1, 0.1), color = 'red', fontsize = 12)
ax.annotate(r'$(a^0, b^0)$', (-1.1, -0.55), color = 'red', fontsize = 12)
ax.annotate(r'$a^1$', (0.7, 0.1), color = 'green', fontsize = 12)
ax.annotate(r'$(a^1, b^1)$', (0.5, 1), color = 'green', fontsize = 12)
ax.annotate(r'$a^2$', (0.25, 0), color = 'purple', fontsize = 12)
ax.annotate(r'$(a^2, b^2)$', (0.05, 0.3), color = 'purple', fontsize = 12)
ax.annotate(r'$(0, 0)$', (-0.15, 0.05), color = 'blue', fontsize = 12)
# uncoment to save image
#plt.savefig('q12.png', bbox_inches='tight')
plt.show()


# Question 2. Nonconvex optimization

# Q2.1

# objective function
f21 = lambda x : np.square(1 - x[0] + x[0]*x[1]) + np.square(2 - x[0] + np.square(x[0])*x[1]) + np.square(3 - x[0] + (x[0]**3)*x[1])
# solve problem
sol21 = minimize(f21, x0 = [0,0], bounds = [(-5,5),(-5,5)])
print('Problem 2.1 solution:')
print(sol21)

# plotting function from Q2.1
# set up grid
x = np.arange(-5.5,5.5,0.1)
y = np.arange(-5.5,5.5,0.1)
X,Y = meshgrid(x, y)
# evaluate function on the grid
Z = f21([X,Y]) 
# in 2D
fig = plt.figure(figsize = (8, 6))
# set x and y limits
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# 2d contour plot
plt.scatter(X, Y, c = Z)
cbar = plt.colorbar()
plt.set_cmap('rainbow')
# limiting color range helps visualize location of optimal solution
plt.clim(0, 10) 
#uncoment to save figure
#plt.savefig('q212d.png')
plt.show()
# in 3D

fig = plt.figure(figsize = (9, 9))
ax = fig.gca(projection = '3d')
# 3d plot
surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.rainbow, linewidth = 0)
fig.colorbar(surf, shrink = 0.5, aspect = 5)
# uncoment to save figure
#plt.savefig('q213d.png')
plt.show()


# Q2.2

# objective function
f22 = lambda x : -0.0001 * (np.abs(np.sin(x[0])*np.sin(x[1])*np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2)/np.pi))) + 1)**0.1
# solving using wild (and quite bad) guess for starting point
sol22 = minimize(f22, x0=[0,0], bounds=[(-10,10),(-10,10)])
print('Problem 2.2 solution:')
print(sol22)

# plotting function from Q2.2
# set grid
x = np.arange(-10.5, 10.5, 0.1)
y = np.arange(-10.5, 10.5, 0.1)
X,Y = meshgrid(x, y) 
# evaluate function on the grid
Z = f22([X,Y])
# in 2D
fig = plt.figure(figsize = (9, 9))
# set x and y limits
plt.xlim(-10, 10)
plt.ylim(-10, 10)
# 2d plot
plt.scatter(X, Y, c=Z)
cbar = plt.colorbar()
plt.set_cmap('rainbow')
# uncoment to save image
#plt.savefig('q222d.png')
plt.show()
# in 3D
fig = plt.figure(figsize = (9, 9))
ax = fig.gca(projection = '3d')
# 3d plot
surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.rainbow, linewidth = 0)
fig.colorbar(surf, shrink = 0.5, aspect = 5)
# uncoment to save image
#plt.savefig('q223d.png')
plt.show()

# first global solution
sol22glob1 = minimize(f22, x0 = [-1,-1], bounds=[(-10,10),(-10,10)])
print('First global solution for problem 2.2:')
print(sol22glob1)
# second global solution
sol22glob2 = minimize(f22, x0 = [1,1], bounds=[(-10,10),(-10,10)])
print('Second global solution for problem 2.2:')
print(sol22glob2)
# third global solution
sol22glob3 = minimize(f22, x0 = [-1,1], bounds=[(-10,10),(-10,10)])
print('Third global solution for problem 2.2:')
print(sol22glob3)
# fourth global solution
sol22glob4 = minimize(f22, x0 = [1,-1], bounds=[(-10,10),(-10,10)])
print('Fourth global solution for problem 2.2:')
print(sol22glob4)

# first local solution 
sol22loc1 = minimize(f22, x0 = [-5,-1], bounds=[(-10,10),(-10,10)])
print('First local solution for problem 2.2:')
print(sol22loc1)
# second local solution
sol22loc2 = minimize(f22, x0 = [7.5,-8.5], bounds=[(-10,10),(-10,10)])
print('Second local solution for problem 2.2:')
print(sol22loc2)