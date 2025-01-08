"""
Module: VorominkEstimation
Author: Dominik Pabst

This module estimates Voronoi and Minkowski tensors based on a given data set.
It includes two main functions:
    `Voromink` for general tensor estimation by solving a least squares problem
    `Vorosurf` for surface Minkowski tensor estimation
Both methods have customizable parameters for precision and algorithm control.

A tensor of rank r is a linear mapping T, which takes r vectors from R^d.
A symmetric tensor T is determined by the values T(e_i1,...,e_ir), where
1 <= i1 <= ... <= ir <= d and e_i is the i-th standard vector in R^d.
In this code a tensor is represented by a dictionary, which has a key (i1,...,ir) for each choice of i1,...,ir.
For example the value corresponding to the key (1,1,2) represents T(e1,e1,e2).

This code is based on the following paper:
    D. Hug, M.A.Klatt, D.Pabst. Minkowski tensors for voxelized data: robust asymptotically unbiased estimators.
Please cite the paper if you use the code.
"""


import math
import numpy as np
import pandas as pd
import itertools as it
import sys
import argparse
import time
from copy import deepcopy

import Tools
import Tensors





def Voromink(infile,r,s,n,Rmax,window=None,a=None,verbose=True,rotate=False):
    
    """
    Estimate the Minkowski tensors of a set represented by a finite data set.
    This approach is consistent for sets with positive reach.

    Args:
        infile (csv or numpy.ndarray): Input data
        r, s (int): Rank parameters of the tensor, which will be estimated.
        n (int): Number of radii, where the Voronoi tensors are evaluated.
        Rmax (float or False): Maximum radius for which the Voronoi tensors are 
                               evaluated. If False, the `window` parameter must be provided 
                               to compute `Rmax`.
        window (list, optional): A cuboid representing the observation window where the 
                                  data lies. If `Rmax` is False, this window is used to 
                                  compute `Rmax`. Example: window = [-2, 2, 0, 1] represents 
                                  the cuboid [-2, 2] x [0, 1].
        a (float, optional): Average nearest neighbor distance in the input data. 
                              If False, the algorithm computes this quantity automatically.
        verbose (bool, optional): If True, the algorithm prints information about the 
                                  current step.
        rotate (bool, optional): If True, the grid process will be rotated randomly.
                                  This option is only necessary for generated test data, which 
                                  lies parallel to the standard axis. Note that rotation 
                                  increases computation time.

    Returns:
        dict: A dictionary with the following keys:
              - "Vor": Contains a list with the estimated Voronoi tensors.
              - "Min": Contains a list with the estimated Minkowski tensors Phi_d,...,Phi_0 (in this order)

    Example usage:
        result = Voromink('data.csv', 50, 0, 2, False, window=[-2, 2, 0, 1])
        print(result['Min'])    # Prints the estimated Minkowski tensors
    """
    
    if Rmax == False and window == None:
        print('either the maximal radius Rmax or the observation window must be given')
        return None

    #print(f"read in {args.infile}")
    if not isinstance(infile, np.ndarray):
        pandadata = pd.read_csv(infile, delimiter = ",", header=None, comment='#')
        in_data = pandadata.values
    else:
        in_data = infile
    
    if verbose == True:
        print("\nData contains "+str(len(in_data))+" points")
        
    if len(in_data) == 0:
        if verbose == True:
            print("\n Input Data is emtpy")
        return 0
    dim = len(in_data[0])   # dimension of the data
    
    
    if window == None:
        window = Tools.create_window(in_data,Rmax)
    
    
    '''reshape observation window'''
    _custom_assert(len(window) == 2*dim,
            f'Error: the number of window values given ({len(window)})'
            +f' is not twice the dimension ({dim});'
            +f' see help for more information')
    window = np.array(window).reshape(dim,2)
    
    if verbose == True:
        print('\nWindow: \n',window)

    
    if Rmax == False:
        '''distance between data & boundary of the observation window'''
        if verbose == True:
            print("\n Compute the distance between data & boundary of the window")
        dist = Tools.distance_window(in_data,window)
        if verbose == True:
            print("It is " + str(round(dist,2)))
    else:
        dist = Rmax
    
    if a == None:
        '''average NN-distance in data'''
        if verbose == True:
            print("\n Compute average NN-distance in data")
        avg = Tools.average_NN(in_data)
        if verbose == True:
            print("It is " + str(round(avg,4)))
    else:
        avg = a


    '''create the grid point process eta'''
    if verbose == True:
        print("\n Construct the grid process eta")
        tic = time.perf_counter()
    if rotate == True:
        eta = Tools.grid_process_rotated(window,avg)
    else:
        eta0 = Tools.grid_process(window,avg)
        eta = []
        for p in it.product(*eta0):
            eta.append(p)
    if verbose == True:
        toc = time.perf_counter()
        print("\nIt took "+Tools._timerep(toc-tic)+" to construct eta")

    
    

    '''compute the n values of the R_i's'''
    R = Tools.R_values(avg,dist,n)
    _custom_assert( not R=="F", "\n The distance to the boundary of the window is too low \n")

    points = len(eta)
    
    
    '''Voronoi-measures'''
    if verbose == True:
        print("\n\n\n Compute the Voronoi-measures (" + str(points) + " iterations)")
        tic = time.perf_counter()
    V = Tensors.Tvoronoi_measure(in_data,eta,R,avg,r,s)
    if verbose == True:
        toc = time.perf_counter()
        print("\nIt took "+Tools._timerep(toc-tic)+" to compute the Voronoi measures")
     
    
    
    ''' Minkowski-Tensors'''
    if verbose == True:
        print("\n\n\n Compute the associated ("+str(r)+","+str(s)+")-Minkowski-Tensors")   
    Phi = Tensors.get_tensors(V,R,dim,r,s)
    
    if verbose == True:
        print("\nTheir values are ")
        Phideep = deepcopy(Phi)
        for k in range(len(Phi)):
            for key in Phideep[k].keys():
                Phideep[k][key] = round(Phideep[k][key],4)
            print("\nPhi_"+str(dim-k)+5*" ", Phideep[k])
            
    #print("\n")
    Results = { "Vor":V, "Min":Phi}
    
    return Results




def Vorosurf(infile,r,s,epsilon,a=None,verbose=True,rotate=False):
    
    """
    Estimate the surface Minkowski tensors of a set represented by a finite data set.
    This approach is consistent for finite unions of compact sets with positive reach.
    Works for s>0 and r=s=0. In the latter the tensor is optained as the trace of the tensor for r=0, s=2 times 1/(4*pi).

    Args:
        infile (csv or numpy.ndarray): Input data
        r, s (int): Rank parameters of the tensor to be estimated.
        epsilon (float): Value of the radius, where the Voronoi tensor is evaluated.
                         We recommend this value to be at least 100 times a.
        a (float, optional): Resolution of the grid. If None, the algorithm 
                             computes the average nearest neighbor distance in the 
                             input data and uses it for resolution.
        verbose (bool, optional): If True, the algorithm prints information about 
                                  the current step.
        rotate (bool, optional): If True, the grid process will be rotated randomly once. 
                                  This is only necessary for generated test data, which 
                                  lies parallel to the standard Cartesian axis.

    Returns:
        dict: Estimated Minkowski tensor.

    Example usage:
        result = Vorosurf('data.csv', 0, 2, 0.1, a=0.001)
        print(result)    # Prints the estimated Minkowski surface tensor
    """
    
    Rmax = epsilon
    
    if not isinstance(infile, np.ndarray):
        pandadata = pd.read_csv(infile, delimiter = ",", header=None, comment='#')
        in_data = pandadata.values
    else:
        in_data = infile

    dim = len(in_data[0]) 
    window = Tools.create_window(in_data,Rmax) 
    
    '''reshape observation window'''
    _custom_assert(len(window) == 2*dim,
            f'Error: the number of window values given ({len(window)})'
            +f' is not twice the dimension ({dim});'
            +f' see help for more information')
    window = np.array(window).reshape(dim,2)
    
    if a == None:
        '''average NN-distance in data'''
        if verbose == True:
            print("\n Compute average NN-distance in data")
        a = Tools.average_NN(in_data)
        if verbose == True:
            print("It is " + str(round(a,4)))
             
    '''create the grid point process eta'''
    if verbose == True:
        print("\n Construct the grid process eta")
        tic = time.perf_counter()
    if rotate == True:
        eta = Tools.grid_process_rotated(window,a)
    else:
        eta0 = Tools.grid_process(window,a)
        eta = []
        for p in it.product(*eta0):
            eta.append(p)
    if verbose == True:
        toc = time.perf_counter()
        print("\nIt took "+Tools._timerep(toc-tic)+" to construct eta")
            
    points = len(eta)
            
    
    if verbose == True:
        print("\n\n\n Compute the Voronoi tensores (" + str(points) + " iterations)")
        tic = time.perf_counter()
    if r==0 and s==0:
        V = Tensors.Tvoronoi_measure(in_data,eta,[epsilon],a,r,2)[0]
        factor =  1 / (2*Tools.kappa(3)*epsilon**(3))
    else:
        V = Tensors.Tvoronoi_measure(in_data,eta,[epsilon],a,r,s)[0]
        factor =  1 / (math.factorial(r)*math.factorial(s)*Tools.kappa(s+1)*epsilon**(1+s))
    if verbose == True:
        toc = time.perf_counter()
        print("\nIt took "+Tools._timerep(toc-tic)+" to compute the Voronoi measures")
    
    for e in V:            
        V[e] *= factor
    
    if r==0 and s==0:
        Phi = Tensors.emptyTensor(dim,0)
        tr = 0
        for j in range(1,dim+1):
            tr += V[(j,j)]
        for e in Phi:
            Phi[e] = tr*4*math.pi
        V = Phi
        
    if verbose == True:
        print("\n The final value is")
        print(V)
              
    return V
    

        
    

def _parse():
    # Parse parameters from the command line
    des = "Voronoi-based estimation of Minkowski tensors in arbitrary dimensions"
    parser = argparse.ArgumentParser(description=des,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Method argument (True or False)
    parser.add_argument('method', type=int, choices=[0, 1], 
                        help='Type in 1 for the algorithm solving a least squares problem or 0 for the alternative approach')

    # Gemeinsame Argumente
    parser.add_argument('r', type=int, action='store',
                        help='First parameter of the Tensors of rank r+s')
    parser.add_argument('s', type=int, action='store',
                        help='Second parameter of the Tensors of rank r+s')

    # Argumente für infile, das immer eingelesen wird
    parser.add_argument('infile', type=str, action='store', 
                        help='Filename of csv file containing the coordinates')

    # Argumente für n und window/Rn nur, wenn method = 1
    parser.add_argument('-n','--n', type=int, action='store', nargs='?', 
                        help='Number of the values in which the Voronoi measure is evaluated (only needed if method is True)')
    parser.add_argument('-w', '--window', type=float, nargs='+', action='store', 
                        help="Coordinates of the window in each dimension (only needed if method is True)\n"
                             "Should be a set of twice the dimension many floats.")
    parser.add_argument('-Rn', '--Rn', type=float, action='store', 
                        help="Alternative to window: Coordinate of Rn for Voronoi measure evaluation (only needed if method is True)")


    # Falls method False ist, diese Argumente einlesen
    parser.add_argument('-epsilon','--epsilon', type=float, action='store', nargs='?', 
                        help='Volume of the parallel set with parameter epsilon is computed (only needed if method is False)')
    parser.add_argument('-a','--a', type=float, action='store', nargs='?', 
                        help='Resolution of the grid used (only needed if method is False)')

    # Parse arguments
    args = parser.parse_args()

    # Logik je nach Wert von method
    if args.method == 1:  # method True
        if args.n is None:
            parser.error('n must be specified when method is True.')
        
        # Überprüfen, ob sowohl window als auch Rn nicht gleichzeitig angegeben wurden
        if args.window and args.Rn:
            parser.error('You cannot specify both --window and --Rn at the same time.')
        if not args.window and not args.Rn:
            parser.error('You must specify either --window or --Rn when method is True.')


    return args









def _custom_assert(boolean, message):
    try:
        assert boolean
    except AssertionError as error:
        print(f'\033[91m{message}\033[0m',
              file=sys.stderr)
        sys.exit(-1)



if __name__ == '__main__':
    
    args = _parse()
    Data = np.loadtxt(args.infile, delimiter=",")
    
    print(args.r)
    print(args.s)
    print(args.method)
    print(args.n)
    print(args.Rn)
    print(args.epsilon)

    if args.method == True:
        if args.Rn == False:
            Voromink(Data, args.r, args.s, args.n, False, args.window)
        else:
            Voromink(Data, args.r, args.s, args.n, args.Rn)
    if args.method == False:

        Vorosurf(Data, args.r, args.s, args.epsilon, args.a)


