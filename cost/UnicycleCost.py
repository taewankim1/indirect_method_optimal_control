from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cvxpy as cp

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from cost import OptimalcontrolCost

class unicycle(OptimalcontrolCost):
    def __init__(self,name,ix,iu,x_t,N):
        super().__init__(name,ix,iu,N)
       
        self.Q = 1e-1*np.identity(3)
        # self.Q = 1e-1 * self.Q
        self.Q[2,2] = 1e-2 * self.Q[2,2]

        self.R = 1 * np.identity(2)
        
        # self.ix = 3
        # self.iu = 2
        self.x_t = x_t
        # self.N = N
        
    def estimate_cost(self,x,u):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
            
        x_diff = np.copy(x)
        x_diff[:,0] = x_diff[:,0] - self.x_t[0]
        x_diff[:,1] = x_diff[:,1] - self.x_t[1]
        x_diff[:,2] = x_diff[:,2] - self.x_t[2]

        x_mat = np.expand_dims(x_diff,2)
        Q_mat = np.tile(self.Q,(N,1,1))
        lx = np.squeeze(np.matmul(np.matmul(np.transpose(x_mat,(0,2,1)),Q_mat),x_mat))
        
        # cost for input
        u_mat = np.expand_dims(u,axis=2)
        R_mat = np.tile(self.R,(N,1,1))
        lu = np.squeeze( np.matmul(np.matmul(np.transpose(u_mat,(0,2,1)),R_mat),u_mat) )
        
        cost_total = 0.5*(lx + lu)
        
        return cost_total