"""
Module: Tensors
Author: Dominik Pabst

This module provides some tools for handeling tensors for the module VorominkEstimation

A tensor of rank r is a linear mapping T, which takes r vectors from R^d.
A symmetric tensor T is determined by the values T(e_i1,...,e_ir), where
1 <= i1 <= ... <= ir <= d and e_i is the i-th standard vector in R^d.
In this code a tensor is represented by a dictionary, which has a key (i1,...,ir) for each choice of i1,...,ir.
For example the value corresponding to the key (1,1,2) represents T(e1,e1,e2).
"""  


import numpy as np
import math
import itertools as it

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import lsq_linear
from sklearn.neighbors import KDTree
from numba import njit
from scipy.special import gamma

import Tools






def Tvoronoi_measure(in_data,eta,R,res,r,s):
    
    """
    Estimates the Voronoi tensors with rank parameters r,s  of the input data for different radii

    Args:
        in_data (numpy.ndarray): Input data
        eta (list): Grid process (entries are the individual points)
        R (list): Entries are the radii, for which the Voronoi tensor is estimated
        res (float): Resolution of the grid process (compare Tools.grid_process)
        r,s (int): Rank parameters of the tensors, that will be estimated

    Returns:
        list: Entries are the estimated Voronoi tensors. Therefore the list has the same length as the input R.
    """
    
    if r+s==0 or r+s >= 5:
        return Tvoronoi_measure_1(in_data,eta,R,res,r,s)
    else:
        return Tvoronoi_measure_2(in_data,eta,R,res,r,s)



def Tvoronoi_measure_1 (in_data,eta,R,res,r,s):
    
    """
    Help function for Tvoronoi_measure
    """
    
    dim = len(in_data[0])
    
    V = [ emptyTensor(dim, r+s) for k in R ]
    
    tree = KDTree(in_data, leaf_size=2)
    A = tree.query(eta,k=1)
    
    jobs = 8         
    K = Parallel(n_jobs=8)(delayed(tensorsum1)(A[0][ math.floor( (i/jobs)*len(A[0])) : math.floor( ((i+1)/jobs)*len(A[0])) ], A[1][ math.floor( (i/jobs)*len(A[0])) : math.floor( ((i+1)/jobs)*len(A[0])) ], in_data, eta[ math.floor( (i/jobs)*len(A[0])) : math.floor( ((i+1)/jobs)*len(A[0])) ],r, s, R) for i in range(jobs))
    
    for Psi in K:
        for k in range(len(R)):
            for e in Psi[k]:
                V[k][e] += Psi[k][e]
    
    for T in V:
        for e in T:
            T[e] *= (res**dim) / (math.factorial(r+s))
    
    return V




def tensorsum1(dist,ind,in_data,eta,r,s,R):
    
    """
    Help function for Tvoronoi_measure
    """
    
    Phix = [ emptyTensor(len(in_data[0]), r+s) for k in R ]
    
    for b in range(len(dist)):
        index = None
        for k in range(len(R)):
            if dist[b][0] <= R[k]:
                index = k
                break
        if index != None:
            for e in Phix[0]:
                z = tensorproduct1(in_data[ind[b][0]], eta[b]-in_data[ind[b][0]], e, r, s)  
                for k in range(index,len(R)):
                    Phix[k][e] += z
                    
    return Phix




def tensorproduct1(x,y,e,r,s):
    
    """
    Help function for Tvoronoi_measure
    """
    
    z = 0
    for sg in it.permutations(range(r+s)):
       add = 1
       for j in range(r+s):
           ind = e[sg[j]]-1
           if j < r:
               add *= x[ind]
           else:
               add *= y[ind]
       z += add

    return z




def Tvoronoi_measure_2 (in_data,eta,R,res,r,s):
    
    """
    Help function for Tvoronoi_measure
    """
    
    dim = len(in_data[0])
    
    V = [ emptyTensor(dim, r+s) for k in R ]
    
    tree = KDTree(in_data, leaf_size=2)
    A = tree.query(eta,k=1)
    
    jobs = 8          
    K = Parallel(n_jobs=8)(delayed(tensorsum2)(A[0][ math.floor( (i/jobs)*len(A[0])) : math.floor( ((i+1)/jobs)*len(A[0])) ], A[1][ math.floor( (i/jobs)*len(A[0])) : math.floor( ((i+1)/jobs)*len(A[0])) ], in_data, eta[ math.floor( (i/jobs)*len(A[0])) : math.floor( ((i+1)/jobs)*len(A[0])) ],r, s, R) for i in range(jobs))
    
    for Psi in K:
        for k in range(len(R)):
            for e in Psi[k]:
                V[k][e] += Psi[k][e]
    
    for T in V:
        for e in T:
            T[e] *= (res**dim) / (math.factorial(r+s))
    
    return V


def tensorsum2(dist,ind,in_data,eta,r,s,R):
    
    """
    Help function for Tvoronoi_measure
    """
    
    Phix = [ emptyTensor(len(in_data[0]), r+s) for k in R ]
    
    for b in range(len(dist)):
        index = None
        for k in range(len(R)):
            if dist[b][0] <= R[k]:
                index = k
                break
        if index != None:
            for e in Phix[0]:
                z = tensorproduct2(in_data[ind[b][0]], eta[b]-in_data[ind[b][0]], e, r, s)  
                for k in range(index,len(R)):
                    Phix[k][e] += z
                    
    return Phix

@njit
def tensorproduct2(x,y,e,r,s):
    
    """
    Help function for Tvoronoi_measure
    """
    
    z = 0
    for sg in perm2(r+s):
       add = 1
       for j in range(r+s):
           ind = e[sg[j]]-1
           if j < r:
               add *= x[ind]
           else:
               add *= y[ind]
       z += add

    return z

@njit
def perm2(n):
    
    """
    Help function for Tvoronoi_measure
    """
    
    P = []
    if n == 1:
        P.append( [0] )
    elif n == 2:
        P.append( [0,1] )
        P.append( [1,0] )
    elif n == 3:
        P.append( [0,1,2] )
        P.append( [0,2,1] )
        P.append( [1,0,2] )
        P.append( [1,2,0] )
        P.append( [2,0,1] )
        P.append( [2,1,0] )
    elif n == 4:
        P.append( [0,1,2,3] )
        P.append( [0,1,3,2] )
        P.append( [0,2,1,3] )
        P.append( [0,2,3,1] )
        P.append( [0,3,1,2] )
        P.append( [0,3,2,1] )
        P.append( [1,0,2,3] )
        P.append( [1,0,3,2] )
        P.append( [1,2,0,3] )
        P.append( [1,2,3,0] )
        P.append( [1,3,0,2] )
        P.append( [1,3,2,0] )
        P.append( [2,0,1,3] )
        P.append( [2,0,3,1] )
        P.append( [2,1,0,3] )
        P.append( [2,1,3,0] )
        P.append( [2,3,0,1] )
        P.append( [2,3,1,0] )
        P.append( [3,0,1,2] )
        P.append( [3,0,2,1] )
        P.append( [3,1,0,2] )
        P.append( [3,1,2,0] )
        P.append( [3,2,0,1] )
        P.append( [3,2,1,0] )
    return P






def get_tensors(V,R,dim,r,s):
    
    """
    Computes the Minkowski tensors from the Voronoi tensors by solving a Least Squares Problem

    Args:
        V (list): Entries are the Voronoi tensors (size should be at leat dim+1)
        R (list): Corresponding radii to the Voronoi tensors in V
        dim (int): Dimension of the set (resp. the data) for which the Voronoi tensors are given
        r,s (int): Rank parameters of the Voronoi tensors

    Returns:
        list: Entries are the computed Minkowski tensors Phi_d,...,Phi_0 (in this order)
    """
    
    A = []
    for v in R:
        Rv = np.array([Tools.kappa(n) * (v**n) for n in range(s,s+dim+1)])
        A.append(Rv)
    A = np.array(A) * math.factorial(r) * math.factorial(s)
    
    Phi = [ emptyTensor(dim, r+s) for d in range(dim+1)]

    for e in Phi[0]:
        b = [ T[e] for T in V ]
        x = lsq_linear(A=A,b=b).x
        for k in range(dim+1):
            Phi[k][e] = x[k]
            
    return Phi




def emptyTensor (dim,rank,LIST=False):
    
    """
    Creates an empty tensor (a tensor whose entries are all zero)

    Args:
        dim (int): Dimension of the tensor (Minkowski tensor of a set in R^d has dimension d)
        rank (int): Rank of the tensor (Minkowski or Voronoi tensor with rank parameters r,s has rank r+s)
        LIST (boolean,optional): If True the entries of the tensor are not 0, but empty lists.
                                 Can be used to bundle several tensors.                                 

    Returns:
        dict: Empty tensor
    """
    
    Phi = {}
    E = [j for j in range(1,dim+1)]
    
    for p in it.product(E, repeat=rank):
        p = tuple(sorted(p))
        if LIST == False:
            Phi[p] = 0
        else:
            Phi[p] = []
    return Phi




def evaluate (T,x):
    
    """
    Computes the values of a tensor at a given argument

    Args:
        T (dict): Tensor, which shall be evaluated
        x (list): Entries are points in R^d, where d is the dimension of t. Length has to be the rank of T.                                 

    Returns:
        float: Computed value
    """
    
    Tx = 0
    rank = len(x)
    dim = len(x[0])
    E = [j for j in range(1,dim+1)]
    for e in it.product(E, repeat=rank):
        add = T[tuple(sorted(e))]
        for i in range(len(e)):
            add *= x[i][e[i]]
        Tx += add

    return Tx






