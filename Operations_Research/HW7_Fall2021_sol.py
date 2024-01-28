import numpy as np
import matplotlib.pyplot as plt
import itertools

# Question 3(a) - plotting feasible region
# construct lines
x = np.linspace(-3, 5, 2000)
# x_2 = 0
y1 = 0*x
# x_1 + x_2 = 4
y2 = 4 - x
# -2 x_1 + x_2 = 2
y3 = 2 + 2*x
# 2 x_1 + x_2 = 6
y4 = 6 - 2*x

# make plot - draw the lines and corner points
fig = plt.figure(figsize = (15, 9))
ax = fig.add_subplot()
plt.plot([0,0],[-2,7],label = r'$x_1=0$', zorder = 1, color = 'dodgerblue')
plt.plot(x, y1, label = r'$x_2=0$', zorder = 2, color = 'deepskyblue')
plt.plot(x, y2, label = r'$x_1+x_2=4$', zorder = 3, color = 'darkblue')
plt.plot(x, y3, label = r'$-2x_1+x_2=2$', zorder = 4, color = 'darkviolet')
plt.plot(x, y4, label = r'$2x_1+x_2=6$', zorder = 5, color = 'darkmagenta')
# set axis limits
plt.xlim((-2, 5))
plt.ylim((-1, 7))
# label axis
ax.annotate(r'$x_2$', (-2, 7), (-2.2, 6.8), color = 'black', fontsize = 12)
ax.annotate(r'$x_1$', (5, 0), (4.8, -1.2), color = 'black', fontsize = 12)

# fill feasible region
plt.fill([0, 3, 2, 2/3, 0], [0, 0, 2, 10/3, 2], color = 'lavender')
plt.legend(bbox_to_anchor = (1.005, 1), loc = 2, borderaxespad=0.)
#plt.show()
plt.savefig("Q3(a).jpg", dpi = 600)


# Question 3(c) Finding basic solutions
c = np.array([-1, -2, 0, 0, 0])
A = np.array([[1, 1, 1, 0, 0], [-2, 1, 0, 1, 0], [2, 1, 0, 0, 1]])
b = np.array([4, 2, 6])
# finding all posible combinations of columns for basis matrix
comb = [list(i) for i in itertools.combinations(range(5), 3)]
# list to store basic feasible solutions
bfs = []
# lists to store costs associated with basic feasible solutions
costb = []
bfsind = []
nonfind = []
for i in range(len(comb)):
    # current basis matrix
    B = A[:, comb[i]]
    # indices of current non-basic variables
    nb = [j for j in range(5) if j not in comb[i]]
    if np.linalg.det(B) != 0:
        # current basic cost
        costb.append(c[comb[i]])
        # current basic solution
        x = np.dot(np.linalg.inv(B), b)
        # if it is feasible store it
        if all(v >= 0 for v in x):
            bfs.append(x)
            costb.append(c[comb[i]])
            bfsind.append(i + 1)
        else:
            nonfind.append(i + 1)
        print("Solution " + str(i+1) + ":")
        print("B = [A_" + str(comb[i][0] + 1) + ", A_" + str(comb[i][1] + 1) + ", A_" + str(comb[i][2] + 1) + "] = ")
        print(B)
        print("x_B = [x_" + str(comb[i][0] + 1) + ", x_" + str(comb[i][1] + 1) + ", x_" + str(comb[i][2] + 1) + "] = ", x)
        print("x_NB = [x_" + str(nb[0] + 1) + ", x_" + str(nb[1] + 1) + "] = ", np.array([0, 0]))
        print("c_B = ", c[comb[i]])
        print("c_NB = ", c[nb])
    else:
        print("Solution " + str(i + 1) + " does not exist, because corresponding basis matrix B = [A_" + str(comb[i][0] + 1) + ", A_" + str(comb[i][1] + 1) + ", A_" + str(comb[i][2] + 1) + "] is singular.")
        
        
# Question 3(d) which solutions are basic feasible solutions
print("Solutions " + ", ".join(str(ind) for ind in bfsind) + " are basic feasible solutions.")
# Question 3(d) which solutions are basic feasible solutions
print("Solutions " + ", ".join(str(ind) for ind in nonfind) + " are infeasible.")
# Question 3(d) adding basic solutions to plot from Q3(a)
# construct lines
x = np.linspace(-3, 5, 2000)
# x_2 >= 0
y1 = 0*x
# x_1 + x_2 <= 4
y2 = 4 - x
# -2 x_1 + x_2 <= 2
y3 = 2 + 2*x
# 2 x_1 + x_2 <= 6
y4 = 6 - 2*x

# make plot - draw lines and label basic solutions
fig = plt.figure(figsize = (15, 9))
ax = fig.add_subplot()
# plot basic solutions
plt.scatter([0, 3, 2, 2/3, 0], [0, 0, 2, 10/3, 2], color = 'green', marker = 'o', zorder = 7, edgecolors = 'black')
plt.scatter([1, -1, 4, 0, 0], [4, 0, 0, 6, 4], color = 'crimson', marker = 'o', zorder = 6, edgecolors = 'black')
# plot lines
plt.plot([0,0],[-4,85],label = r'$x_1=0$', zorder = 1, color = 'dodgerblue')
plt.plot(x, y1, label = r'$x_2=0$', zorder = 2, color = 'deepskyblue')
plt.plot(x, y2, label = r'$x_1+x_2=4$', zorder = 3, color = 'darkblue')
plt.plot(x, y3, label = r'$-2x_1+x_2=2$', zorder = 4, color = 'darkviolet')
plt.plot(x, y4, label = r'$2x_1+x_2=6$', zorder = 5, color = 'darkmagenta')
# set axis limits
plt.xlim((-2, 5))
plt.ylim((-1, 7))
# label axis
ax.annotate(r'$x_2$', (-2, 7), (-2.2, 6.8), color = 'black', fontsize = 12)
ax.annotate(r'$x_1$', (5, 0), (4.8, -1.2), color = 'black', fontsize = 12)
# label points
ax.annotate(r'$b1 = (1, 4)$', (1, 4), (1.1, 4.0), color = 'crimson', fontsize = 12)
ax.annotate(r'$(b2) = (2, 2)$', (2, 2), (2.05, 2.05), color = 'green', fontsize = 12)
ax.annotate(r'$(b3) = \left(\frac{2}{3}, \frac{10}{3}\right)$', (2/3, 10/3), (0.35, 3.6), color = 'green', fontsize = 12)
ax.annotate(r'$(b4) = (3, 0)$', (3, 0), (3.1, -0.2), color = 'green', fontsize = 12)
ax.annotate(r'$b5 = (-1, 0)$', (-1, 0), (-0.95, -0.15), color = 'crimson', fontsize = 12)
ax.annotate(r'$b6 = (4, 0)$', (4, 0), (4.05, 0.05), color = 'crimson', fontsize = 12)
ax.annotate(r'$b7 = (0, 6)$', (0, 6), (0.05, 5.95), color = 'crimson', fontsize = 12)
ax.annotate(r'$(b8) = (0, 2)$', (0, 2), (0.05, 2.0), color = 'green', fontsize = 12)
ax.annotate(r'$b9 = (0, 4)$', (0, 4), (0.05, 4.05), color = 'crimson', fontsize = 12)
ax.annotate(r'$(b10) = (0, 0)$', (0, 0), (0.05, 0.05), color = 'green', fontsize = 12)

# fill feasible region
plt.fill([0, 3, 2, 2/3, 0], [0, 0, 2, 10/3, 2], color = 'lavender')
plt.legend(bbox_to_anchor = (1.005, 1), loc = 2, borderaxespad = 0.)
#plt.show()
plt.savefig("Q3(d).jpg", dpi = 600)