#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:35:41 2018

@author: nature
"""
from scipy.optimize import minimize

import numpy as np
import numdifftools as nd
from sklearn.metrics.pairwise import pairwise_kernels

a=np.array([[1,2],[3,4],[2,2]])
b = np.array([[1,1],[2,1],[4,1]])
c = np.array([1,2])
d=np.array([[2,3],[1,2],[4,1]])
print np.sum(c**2,axis=1)
#a_norm = np.sum(a ** 2, axis = -1)
#b_norm = np.sum(b ** 2, axis = -1)
#gamma = 1e3
#K = np.exp(-gamma * (a_norm[:,None] + b_norm[None,:] - 2 * np.matmul(a, np.transpose(b))))
#print K
#k2 = np.array([1,2,3])
#k3 = np.array([1,1,2])
#C_group = np.array([1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4])
##k4 = np.multiply(k2,k3)
#k5 = np.array([[2,4],[1,2],[1,1]])
#print np.sum(k5)
#k6 = np.transpose(k5)
#k7 = np.array([1,2])
#k8 = np.array([[2],[1]])

y = np.array([[1],[-1],[1],[1]])

alphas_init = np.zeros((4,))
K= np.ones((4,4))
def objective(alphas):
    y = np.array([[1],[-1],[1],[1]])
    num_samples, = alphas.shape
    alphas_row = alphas.reshape((1,num_samples))
    y_row = y.reshape((1,num_samples))
    element_alpha = np.matmul(np.transpose(alphas_row),alphas_row)
    element_y = np.matmul(y,y_row)
    element = np.multiply(element_alpha,element_y)
    element = np.multiply(element,K)
    print element
    obj = np.sum(alphas) -1/2*np.sum(element)
    # turn max into minimize
    obj = -obj 
    return obj

#Jfun = nd.Jacobian(objective)
#print Jfun([1,1,1,1])

def gradient(alphas) :
    y = np.array([[1],[-1],[1],[1]])
    num_samples = alphas.shape[0]
    alphas_row = alphas.reshape((1,num_samples))
    y_row = y.reshape((1,num_samples))
    element_y = np.matmul(y,y_row)
    M = np.multiply(element_y,K)
    A = np.matmul(alphas_row,M)
    print A
    gradient = 1 - A
    gradient = -gradient
    return gradient

alpha = np.ones((4,1))


#def constraint1(alphas):
#    num_samples, = alphas.shape
#    res = np.multiply(alphas,y)
#    res = np.sum(res)
#    return res
#
#num_samples, = alphas_init.shape  
#    
#constraints = (
#    {'type': 'eq',
#     'fun': constraint1})
##     'jac': y.reshape((num_samples,))})
#
## Define the bounds for each alpha.
## TODO: implement this.
#
#bounds = ((0,1),)
#for i in range(num_samples - 1) :
#    bounds = bounds + ((0,1),)
#
## Define the initial value for alphas.
#alphas_init = np.zeros((num_samples,))
#
## Solve the QP.
#result = minimize(objective, alphas_init, method="SLSQP", jac=True,
#    bounds=bounds, constraints=constraints, tol=1e-5,
#    options={'ftol': 1e-5, 'disp': 2})
#alphas = result['x']  

#def obj(x):
#    objective = (x[0] - 1)**2 + (x[1] - 2.5)**2
#    gradient = np.array([2*(x[0] - 1),2*(x[1] - 2.5)])
#    return (objective,gradient)
#
#cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},      
#        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
#        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
#
#bnds = ((0, None), (0, None))
#res = minimize(obj, (2, 0), method='SLSQP', jac = True, bounds=bnds,constraints=cons)
#
#print res

#all_1 = np.ones((6,1))
#k5=np.matmul(y_row,all_1)
#dd = np.ones((4,4))/2 + np.eye(4)
#d = 2 - np.diag(dd)

#a = np.array([8,4,2,5,1,3])
#aa = np.array([i for i,v in enumerate(a) if v > 4])
#aaa = np.array([i for i,v in enumerate(a[aa]) if v < 8])