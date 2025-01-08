"""
Module: Tools
Author: Dominik Pabst

This module provides some basic tools for the module VorominkEstimation
"""   



import numpy as np
import math
import itertools as it
import time

from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.neighbors import KDTree
from scipy.special import gamma
from scipy.optimize import lsq_linear
from copy import deepcopy

              



def grid_process(W,res):
    
    """
    Constructs a grid process in an observation window.
    This is a grid with resolution res randomly translated (by a uniformly distributed vector).

    Args:
        window (list): Cuboid representing the observation window where the 
                                  data lies. Example: window = [[-2, 2],[0, 1]] represents 
                                  the cuboid [-2, 2] x [0, 1]
        res (float): Resolution of the grid process.

    Returns:
        list: Entries are lists itself. The j-th list contains the values
              of the j-th coordinates of the points of the grid. Example: [[0,1],[2,3]]
              represents the grid consisting of the 4 points (0,2),(0,3),(1,2),(1,3)
    """
    
    dim = len(W)
    u = np.random.uniform(0, res, size=dim)
    #u = [0 for j in range(dim)]
    koord = []
    for j in range(dim):       
        z = math.floor((W[j][1]-W[j][0])/res)    # number of points in the grid in dimension j
        jkoord = [W[j][0]+u[j]+i*res for i in range(z)]
        koord.append(jkoord)

    return koord



def grid_process_rotated(W,res):
    
    """
    Constructs a randomly rotated grid process in an observation window.

    Args:
        window (list): A cuboid representing the observation window where the 
                                  data lies. Example: window = [[-2, 2],[0, 1]] represents 
                                  the cuboid [-2, 2] x [0, 1]
        res (float): The resolution of the grid process.

    Returns:
        list: The elements of the list are the points of the grid process.
    """
    
    dim = len(W)
    pairs = {}   # pairs of dimensions that will be rotated
    m = 1
    while m < dim:
        pairs[(m,m+1)] = np.random.uniform(0,1)*2*math.pi
        m += 2
    if m == dim:
        pairs[(m-1,m)] = np.random.uniform(0,1)*2*math.pi
        
    k = 0
    for side in W:
        k += 0.25*(side[1]-side[0])**2
    k = math.sqrt(k)
    W0 = [[-k,k] for side in W]
    eta0 = grid_process(W0, res)
    eta = []
    for p in it.product(*eta0):
        z = list(deepcopy(p))
        for fl in pairs:
            x = p[fl[0]-1]
            y = p[fl[1]-1]
            th = pairs[fl]
            z[fl[0]-1] = x*np.cos(th)-y*np.sin(th)
            z[fl[1]-1] = x*np.sin(th)+y*np.cos(th)
        IN = 1
        for j in range(len(W)):
            z[j] += 0.5*(W[j][0]+W[j][1])
            if z[j] < W[j][0] or W[j][1] < z[j]:
                IN = 0
                break
        if IN == 1:
            eta.append(z)
            
    return eta



def create_window(in_data,dist,F=False):
    
    """
    Constructs a cuboid, whose boundary has in each direction distance equal to the parameter dist from the data.

    Args:
        in_data (numpy.ndarray): Input data
        dist (float): Distance of the boundary of the output cuboid to the data
        F (boolean,optional): Decides in which format the cuboid is returned
        

    Returns:
        list: Contains the coordinates of the cuboid.
              Example: [-2,2]x[-2,2]
              if F is True, the algorithm returns [ [-2,2], [-2,2] ]
              if F is False, the algorithm returns [ -2, 2, -2, 2]
    """
    
    dim = len(in_data[0])
    W = [[in_data[0][j],in_data[0][j]] for j in range(dim)]
    for p in in_data:
        for j in range(dim):
            if p[j] < W[j][0]:
                W[j][0] = p[j]
            if p[j] > W[j][1]:
                W[j][1] = p[j]
    for j in W:
        j[0] -= dist
        j[1] += dist
        
    if F == False:
        WR = []
        for d in W:
            for k in range(len(d)):
               WR.append(d[k])
    else:
        WR = W
    return WR




def average_NN(in_data):
    
    """
    Calculates the average nearest neighbour distance in the data.

    Args:
        in_data (numpy.ndarray): Input data   

    Returns:
        float: Average Nearest Neighbour Distance
    """
    
    tree = KDTree(in_data, leaf_size=2)
    dist = tree.query(in_data,k=2)[0]
    
    avg = 0
    for d in dist:
        avg += d[1]
    avg = avg / len(in_data)
    
    return avg




def minimal_NN(in_data):
    
    """
    Calculates the minimal nearest neighbour distance in the data.

    Args:
        in_data (numpy.ndarray): Input data   

    Returns:
        float: Minimal Nearest Neighbour Distance
    """
    
    tree = KDTree(in_data, leaf_size=2)
    dist = tree.query(in_data,k=2)[0]
    D = [ d[1] for d in dist]
    return min(D)





def distance_window(in_data,W):
    
    """
    Calculates the distance of the boundary of the cuboid W to the input data.

    Args:
        in_data (numpy.ndarray): Input data  
        W (list): Cuboid, where the data lies in. Should be of the form like the
                  function create_window creates it with the option F=True

    Returns:
        float: Distance of the boundary of the cuboid W and the data.
    """
    
    dim = len(in_data[0])
    dist = min( abs(in_data[0][0]-W[0][0]), abs(in_data[0][0]-W[0][1]))
    for p in in_data:
        for j in range(dim):
            if abs(p[j]-W[j][0]) < dist:
                dist = abs(p[j]-W[j][0])
            if abs(p[j]-W[j][1]) < dist:
                dist = abs(p[j]-W[j][1])
    return dist




def R_values(Rmin,Rmax,n):
    
    """
    Computes n values equidistant between Rmin,Rmax (those values excluded)

    Args:
        Rmin (float): Lower bound
        Rmax (float): Upper bound
        n (int): Number of values

    Returns:
        list: Contains the n values.
    """
    
    if Rmax-Rmin<=0:
        return "F"
    delta = (Rmax-Rmin) / (n+1)
    R = [Rmin + (j*delta) for j in range(1,n+1)]
    return R
    


def kappa(n):
    
    """
    Computes the n-dimensional volume of the n-dimensional unit ball.

    Args:
        n (int): Dimension
    Returns:
        float: Computed Volume
    """
    
    return (np.pi**(n/2)) / gamma(n/2+1)



def euclidean(x):
    
    """
    Computes Eudlidean norm of a vector x.

    Args:
        x (list): Eclidean vector
    Returns:
        float: Computed norm
    """
    
    y = 0
    for xj in x:
        y += xj**2
    return math.sqrt(y)



def dist(x,y):
    
    """
    Computes Eudlidean distance of two vectors x and y.

    Args:
        x,y (list): Eclidean vectors
    Returns:
        float: Computed distance
    """
    
    d = 0
    for j in range(len(x)):
        d += (x[j]-y[j])**2
    return math.sqrt(d)



def _timerep(sec):
    c = 60*60*24
    day = 0
    while sec >= c:
        sec -= c
        day += 1
    t = time.gmtime(sec)
    t = time.strftime("%H:%M:%S", t)
    
    if day == 0:
        return t
    else:
        return str(day)+" days "+t

        
        
        
        
        
        
        
        