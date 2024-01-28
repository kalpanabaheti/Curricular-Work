
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import numpy as np
import copy
import math


def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    
    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    
    
    m1 = np.array([((X - MU[k]).T)@(X - MU[k]) for k in range(MU.shape[0])])
    matrix = m1/X.shape[0]
        
    testing = np.array(matrix)
    return testing


def initialize_parameters(X, k):
    """
    Returns initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    
    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k 
    """
    MU = X[np.random.choice(X.shape[0], k, replace=False), :]
    SIGMA = compute_sigma(X, MU)
    PI = np.array([1/k]*k)
    
    return MU, SIGMA, PI


def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] (for single datapoint) 
        or numpy.ndarray[numpy.ndarray[float]] (for array of datapoints)
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float (for single datapoint) 
                or numpy.ndarray[float] (for array of datapoints)
    """
    
    s_det = np.linalg.det(sigma)
    pi = math.pi
    e = math.e
    n = 3
    s_inv = np.linalg.inv(sigma)
    
    denominator = ((2*pi)**(n/2))*abs((s_det**(1/2)))
    
    
    x_mu = x - mu
    if len(x.shape)==2:
        
        
        x_mut = np.einsum('ji',x_mu)
        part0 = np.dot(x_mu,s_inv)
        part1 = part0*x_mu
        part2 = np.sum(part1,axis=1)
        
        exponent = (-1/2)*(part2)
        
        prob = (e**exponent)/denominator
        #print('exp calc done')
        
        
        
    else:
        
        x_mut = x_mu.T
        #print('subtract and transpose done')
        part1 = np.matmul(x_mut,s_inv)
    
        part2 = np.matmul(part1,x_mu)
        #print('multiplication done')
    
        exponent = (-1/2)*(part2)
        #print('exp calc done')
        prob = (e**exponent)/denominator
        #print(prob)

       
    return prob


def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation 
    Calculate responsibility for each
    of the data points, for the given 
    MU, SIGMA and PI.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int
    
    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    
    probs = []
    s = 0
    for cluster in range(k):
        p = prob(X, MU[cluster], SIGMA[cluster]) 
        num = PI[cluster]*p
        s+=num
        probs.append(num)
        
    s = np.array(s)
    probs = np.array(probs)
    
    resps = np.array([top/s for top in probs])
    
    return resps



def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int
    
    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    """
    
    new_MU = []
    for row in r:
        
        rowt = row.T
        nk = np.sum(rowt)
        new_row = np.dot(rowt,X)/nk
        new_MU.append(new_row)
        
    new_MU = np.array(new_MU)
    
    rt = r.T
    new_SIGMA = []
    for i in range(new_MU.shape[0]):
        
        row = r[i]
        rowt = row.T
        nk = np.sum(rowt)
        part1 = np.multiply(rowt,(X-new_MU[i]).T)
        new_row = np.dot(part1,(X-new_MU[i]))/nk
        #print(new_row)
        new_SIGMA.append(new_row)
    
    
    new_SIGMA = np.array(new_SIGMA)
    new_PI = np.sum(r.T, axis=0)/(r.T).shape[0]

    return new_MU, new_SIGMA, new_PI



def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the 
    trained model based on the following
    formula for posterior probability:
    
    log(Pr(X | mixing, mean, stdev)) = sum((i=1 to m), log(sum((j=1 to k),
                                      mixing_j * N(x_i | mean_j,stdev_j))))

    Make sure you are using natural log, instead of log base 2 or base 10.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    log_likelihood = float
    """
    
    joint_probs = np.array([PI[i]*prob(X, MU[i], SIGMA[i]) for i in range(k)])
    
    joint_probs = np.sum(joint_probs, axis=0)
    joint_probs = np.log(joint_probs)
    log_likelihood = np.sum(joint_probs)
    
    #print(log_likelihood )
    
    return log_likelihood


def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the 
    expectation-maximization algorithm. 
    E.g., iterate E and M steps from 
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example 
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    
    if not isinstance(initial_values, tuple):
        MU, SIGMA, PI = initialize_parameters(X, k)
        
    else:
        MU, SIGMA, PI = initial_values
        
    log_likelihood = likelihood(X, PI, MU, SIGMA, k)
    
    count = 0
    convergence = False
        
    while not convergence:
        
        # E step
        responsibility = E_step(X,MU,SIGMA,PI,k)
        
        # M step
        MU, SIGMA, PI = M_step(X, responsibility, k)
        
        # Evaluate likelihood and check for convergence
        prev_likelihood = log_likelihood
        log_likelihood = likelihood(X, PI, MU, SIGMA, k)
        
        count, convergence = convergence_function(prev_likelihood, log_likelihood, count)
        
    return MU, SIGMA, PI, responsibility




