
# coding: utf-8

# In[4]:
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
import IPython
class PMP:
    def __init__(self,name,horizon,maxIter,Model,Cost):
        self.name = name
        self.model = Model
        self.cost = Cost
        self.N = horizon
        self.maxIter = maxIter
        
        self.verbosity = True
        self.alpha = 0.3
        self.tol_fun = 1e-7

        # self.dlamda = 1.0
        # self.lamda = 1.0
        # self.lamdaFactor = 1.6
        # self.lamdaMax = 1e10
        # self.lamdaMin = 1e-6
        # self.tolFun = 1e-7
        # self.tolGrad = 1e-4
        # self.zMin = 0
        self.last_head = True
        
        self.initialize()
        
    def initialize(self) :
        
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N,self.model.iu))
        self.cs = np.zeros((self.N+1,self.model.ix))
        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N,self.model.iu))
        self.Alpha = np.power(10,np.linspace(0,-3,11))
        self.du = np.ones((self.N,self.model.iu))

        self.l = np.zeros((self.N,self.model.iu))
        self.L = np.zeros((self.N,self.model.iu,self.model.ix))
        self.fx = np.zeros((self.N,self.model.ix,self.model.ix))
        self.fu = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.c = np.zeros(self.N+1)
        self.cnew = np.zeros(self.N+1)
        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
    
    def forward(self,x0,x,u,du,alpha):
        # TODO - change integral method to odefun
        # horizon
        N = self.N
        
        # x-difference
        dx = np.zeros(self.model.ix)
        
        # variable setting
        xnew = np.zeros((N+1,self.model.ix))
        unew = np.zeros((N,self.model.iu))
        cnew = np.zeros(N+1)
        
        # initial state
        xnew[0,:] = x0
        
        # roll-out
        for i in range(N):
            unew[i,:] = u[i,:] - du[i,:] * alpha
            xnew[i+1,:] = self.model.forward(xnew[i,:],unew[i,:],i)
            cnew[i] = self.cost.estimate_cost(xnew[i,:],unew[i,:])
        cnew[N] = self.cost.estimate_cost(xnew[N,:],np.zeros(self.model.iu))

        return xnew,unew,cnew
        
    def backward(self):
        # TODO - change integral method to odefun
        # horizon
        N = self.N

        # state & input size
        ix = self.model.ix
        iu = self.model.iu
        
        # variable setting
        cs = np.zeros((N+1,self.model.ix))
        
        # final costate
        cs[-1] = self.cx[-1]

        for i in range(self.N-1,-1,-1):
            cs[i] = self.cx[i] + np.matmul(cs[i+1],self.fx[i])
            
        return cs
                   
        
    def update(self,x0,u0):
        # current position
        self.x0 = x0
        
        # initial input
        self.u = u0
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # timer setting
        # trace for iteration
        # timer, counters, constraints
        # timer begin!!
        
        # generate initial trajectory
        diverge = False
        stop = False

        self.x[0,:] = self.x0
        for j in range(np.size(self.Alpha,axis=0)):   
            for i in range(self.N):
                self.x[i+1,:] = self.model.forward(self.x[i,:],self.Alpha[j]*self.u[i,:],i)       
                self.c[i] = self.cost.estimate_cost(self.x[i,:],self.Alpha[j]*self.u[i,:])
                if  np.max( self.x[i+1,:] ) > 1e8 :                
                    diverge = True
                    print("initial trajectory is already diverge")
                    pass
            self.c[self.N] = self.cost.estimate_cost(self.x[self.N,:],np.zeros(self.model.iu))
            if diverge == False:
                break
                pass
            pass

        # iterations starts!!
        flgChange = True
        for iteration in range(self.maxIter) :
            # differentiate dynamics and cost
            if flgChange == True:
                start = time.time()
                self.fx, self.fu = self.model.diff(self.x[0:N,:],self.u)
                c_x_u = self.cost.diff_cost(self.x[0:N,:],self.u)
                self.cx[0:N,:] = c_x_u[:,0:self.model.ix]
                self.cu[0:N,:] = c_x_u[:,self.model.ix:self.model.ix+self.model.iu]
                c_x_u = self.cost.diff_cost(self.x[N:,:],np.zeros((1,iu)))
                self.cx[N,:] = c_x_u[0:self.model.ix]
                flgChange = False
                pass

            time_derivs = (time.time() - start)

            # backward pass
            self.cs = self.backward()

            # update u
            self.du = self.cu + np.squeeze(np.matmul(np.transpose(np.expand_dims(self.cs[1:],2),(0,2,1)),self.fu))

            # step3. line-search to find new control sequence, trajectory, cost
            fwdPassDone = False
            start = time.time()
            for i in self.Alpha :
                self.xnew,self.unew,self.cnew = self.forward(self.x0,self.x,self.u,self.du,i)
                dcost = np.sum( self.c ) - np.sum( self.cnew )
                if dcost > 0 :
                    fwdPassDone = True
                    break          
            if fwdPassDone == False :
                alpha_temp = 1e8 # % signals failure of forward pass
                pass
            time_forward = time.time() - start
                
            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   cost        reduction")#   expected    gradient    log10(lambda)")
                pass

            if fwdPassDone == True:
                if self.verbosity == True:
                    print("%-12d%-12.6g%-12.3g" % ( iteration,np.sum(self.c),dcost) )     

                # accept changes
                self.u = self.unew
                self.x = self.xnew
                self.c = self.cnew
                flgChange = True

                if dcost < self.tol_fun :
                    if self.verbosity == True:
                        print("SUCCEESS: cost change < tolFun",dcost)
                    break
            else : # no cost improvement
                # print status
                if self.verbosity == True :
                    print("%-12d%-12s%-12.3g" %
                        ( iteration,'NO STEP', dcost ))
                break



        return self.x, self.u 
        


        
        
        
        
        
        
        
        
        
        
        
        


