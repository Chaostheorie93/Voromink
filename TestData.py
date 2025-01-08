"""
Module: TestData
Author: Dominik Pabst

With this module some test data can be generated.
"""  


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from tqdm import tqdm

import Tools




def Cuboid(sides,res):
    
    """
    Creates a Cuboid in arbitrary dimension (centered at the origin) (intersected with a finite grid)

    Args:
        sides(list): Entries are the intervals in each dimension.
                     Example: To create the cuboid [-2, 2] x [0, 1] the input has to be [[-2,2],[0,1]]
        res (float): Resolution of the grid, which will be intersected with the cuboid to obtain a finite data set.

    Returns:
        numpy.ndarray: Created data.
    """
    
    C = []
    Coord = []
    for a in sides:
        acoord = [ j*res -0.5*a for j in range(math.floor((a/res)+1)) ]
        Coord.append(acoord)
    for p in it.product(*Coord):#tqdm(it.product(*Coord)):
        C.append(list(p))
    
    return np.array(C)





def roundCub(sides,res,r):
    
    """
    Creates a Cuboid with rounded corners in arbitrary dimension (centered at the origin) (intersected with a finite grid)

    Args:
        sides(list): Entries are the intervals in each dimension (for the inner cuboid)
                     Example: To create the cuboid [-2, 2] x [0, 1] the input has to be [[-2,2],[0,1]]
        res (float): Resolution of the grid, which will be intersected with the cuboid with rounded corners to obtain a finite data set.
        r (float): Radii of the spherical segments at the corners.

    Returns:
        numpy.ndarray: Created data.
    """
    
    sides0 = [ a+2*r for a in sides]
    Cub0 = Cuboid(sides0, res)
    rndCub = []
    for p in Cub0:
        hit = []
        for j in range(len(p)):
            if p[j]>sides[j]*0.5:
                hit.append(1)
            elif p[j]<-sides[j]*0.5:
                hit.append(-1)
            else:
                hit.append(0)
        if 0 in hit:
            rndCub.append(p)
        else:
            z = []
            for j in range(len(p)):
                z.append(hit[j]*(sides[j]*0.5))
            if Tools.dist(p,z) <= r:
                rndCub.append(p)
    return np.array(rndCub)


       


def rotObject(D):
    
    """
    Rotates a given data set

    Args:
        D (numpy.ndarray): Data set, which shall be rotated.

    Returns:
        Nothing (the data given will be rotated)
    """
    
    dim = len(D[0])
    pairs = {}   # pairs of dimensions that will be rotated
    m = 1
    while m < dim:
        pairs[(m,m+1)] = np.random.uniform(0,1)*2*math.pi
        m += 2
    if m == dim:
        pairs[(m-1,m)] = np.random.uniform(0,1)*2*math.pi

    for p in D:
        for d in pairs:
            x = p[d[0]-1]
            y = p[d[1]-1]
            th = pairs[d]
            p[d[0]-1] = x*np.cos(th)-y*np.sin(th)
            p[d[1]-1] = x*np.sin(th)+y*np.cos(th)
            
    


def Ellipsoid2D(a,b,res):
    
    """
    Creates a 2-dimensional ellipsoid (centered at the origin) (intersected with a finite grid)

    Args:
        a (float): 0.5*width of the ellipse
        b (float): 0.5*heigth of the ellipse
        res (float): Resolution of the grid, which will be intersected with the ellipsoid to obtain a finite data set.

    Returns:
        numpy.ndarray: Created data.
    """
    
    W = [[-a-0.5*res,a+0.5*res],[-b-0.5*res,b+0.5*res]]
    Grid = grid(W,res)     # big enough grid
    E = []
    for y in it.product(*Grid):
        p = np.array(y)
        if p[0] == -a and p[1] == 0:
            print("here")
            print(p[0]**2 / a**2 + p[1]**2 / b**2 <= 1)
        if p[0]**2 / a**2 + p[1]**2 / b**2 <= 1:
            E.append(p)
    return np.array(E)



def grid(W,res):
    
    """
    Creates a grid with resolution res inside a cuboid W

    Args:
        W (list): Cuboid
                  Example: The cuboid [-2, 2] x [0, 1] is represented by [[-2,2],[0,1]]
        res (float): Resolution of the grid

    Returns:
        list: j-th entry contains all the coordinates w.r.t. to the j-th dimension.
              To get all the points of the grid, one has to take all combinations of those values.
    """
    
    dim = len(W)
    koord = []
    for j in range(dim):
        # number of points in the grid in dimension j
        z = math.floor((W[j][1]-W[j][0])/res)
        jkoord = [W[j][0]+i*res for i in range(z+1)]
        koord.append(jkoord)
    return koord






def spShell(r1,r2,res):
    
    """
    Creates a spherical shell (a ball, where a smaller ball is removed) (centered at the origin) (intersected with a finite grid)

    Args:
        r1 (float): radius of the inner ball
        r2 (float): radius of the outer ball
        res (float): Resolution of the grid, which will be intersected with the spherical shell to obtain a finite data set.

    Returns:
        numpy.ndarray: Created data.
    """
    
    W = [[-2*r2,2*r2],[-2*r2,2*r2]]
    S = []
    Grid = grid(W, res)
    for y in it.product(*Grid):
        p = np.array(y)
        if Tools.euclidean(p) >= r1 and Tools.euclidean(p) <= r2:
            S.append(p)
    return np.array(S)






def cuttedRect(a1,b1,a2,b2,res):
    
    """
    Creates a rectangle, where a smaller rectangle is removed (centered at the origin) (intersected with a finite grid)

    Args:
        a1,b1 (floats): side lengths of the inner rectangle
        a2,b2 (floats): side lengths of the outer rectangle
        res (float): Resolution of the grid, which will be intersected with the rectangle to obtain a finite data set.

    Returns:
        numpy.ndarray: Created data.
    """
    
    R = Cuboid([a2,b2], res)
    RR = []
    for p in R:
        if p[0] <= -0.5*a1 or p[0] >= 0.5*a1:
            RR.append(p)
        elif p[1] <= -0.5*b1 or p[1] >= 0.5*b1:
                RR.append(p)
    return np.array(RR)




def plotData(D,s,d1=1,d2=2):
    
    """
    Plots given data

    Args:
        D (numpy.ndarray): Data, which shall be plotted
        s (int): size of the points in the plot
        d1,d2 (int, optional): If data has more than 2 dimensions, the dimensions d1 and d2 will be plotted.

    Returns:
        Nothing, but creates a plot.
    """
    
    xcoord = []
    ycoord = []

    for p in D:
        xcoord.append(p[d1-1])
        ycoord.append(p[d2-1])
    plt.scatter(xcoord,ycoord,s=s)
  


        
    
    

if __name__ == '__main__':
    
    
    
    ''' rotated rectangle '''
    a = 5
    b = 3
    res = 0.02
    pixelsize  = 0.1
    C = Cuboid([a, b], res)
    rotObject(C)
    
    plt.figure(figsize=(10,10))
    plt.title("Rotated Rectangle with NN-distance " + str(res))
    plotData(C,pixelsize)
    #plt.annotate("a = " + str(a) + "\nb = " + str(b), xy=(0.9, 0.87), xycoords='axes fraction')
    
    
    
    
    ''' Ellipsiod '''
    a = 2
    b = 2
    res = 0.02
    
    E = Ellipsoid2D(a, b, res)

    plt.figure(figsize=(10,10))
    plt.xlim((-a-0.1,a+0.1))
    plt.ylim((-b-0.1,b+0.1))
    plt.title("Ellipse with NN-distance " + str(res))
    plotData(E,pixelsize)
    #plt.annotate("a = " + str(a) + "\nb = " + str(b), xy=(0.9, 0.87), xycoords='axes fraction')

    

    ''' Rounded Rectangle '''
    a = 8
    b = 5
    res = 0.02
    r = 1
    m = max(a,b)
    RC = roundCub([a,b], res,r)
    
    plt.figure(figsize=(10,10))
    plt.xlim((-0.5*m-r-1,0.5*m+r+1))
    plt.ylim((-0.5*m-r-1,0.5*m+r+1))
    plt.title("Rounded Rectangle with NN-distance " + str(res))
    plotData(RC,pixelsize)
    
    
    
    ''' Spherical Shell '''
    r1 = 1
    r2 = 2
    res = 0.02
    SS = spShell(r1,r2,res)
    
    plt.figure(figsize=(10,10))
    plt.xlim((-1.01*r2,1.01*r2))
    plt.ylim((-1.01*r2,1.01*r2))
    plt.title("Spherical Shell with NN-distance " + str(res))
    plotData(SS,pixelsize)
    
    
    
    ''' Cutted Rect '''
    a1 = 1
    b1 = 2
    a2 = 3
    b2 = 5
    c = max(a2,b2)
    res = 0.02
    SS = cuttedRect(a1, b1, a2, b2, res)
    
    plt.figure(figsize=(10,10))
    plt.xlim((-1.1*0.5*c,1.1*0.5*c))
    plt.ylim((-1.1*0.5*c,1.1*0.5*c))
    plt.title("Cutted Rectangle with NN-distance " + str(res))
    plotData(SS,pixelsize)
    

    
    
    
    
    
    
    
    
    
    
    
    
    