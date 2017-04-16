#==============================================================================
#Implementation of Gillespie's Direct Method for sampling from a Lotka Volterra system
#need to define the propensity function (here propLV)
#outputs:
#xmat : matrix of the updated state for each reaction time
#tmat: matrix of the reaction times
#==============================================================================
import numpy as np
import math
import random
#from propensities import Propensities
from propLV import PropLV
#import matplotlib.pyplot as plt
########## Gillespie - Direct Method #########################

class GillespieLV(object):
    def __init__(self,S,tmax,x,c1,c2,c3):
        self.S = S
        self.tmax = tmax
        self.x = x
        
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)

    def calc_gillespie(self):
        t = 0
        xmat = self.x
        tmat = t

        while t <= self.tmax :
            propensity = PropLV(self.x,t,self.c1,self.c2,self.c3)
            prop = propensity.calculate()
            #print prop
            h0 = np.sum(prop)
            #find next reaction time from an exponential with rate h0(using the inverse transformation method)
            t += -(1/h0)*math.log(random.random())#(1/h0)*math.log(1/random.random())
            #print t
            r = random.random()*h0
            i = 0
            #find next reaction if 0<r<h(1)/h0 reaction 1 is happening, if h(1)/h0<r<[h(1)+h(2)]/h0 then reaction 2 is happening etc.
            while(np.sum(prop[0:i+1]))<r :
                i += 1
            self.x += self.S[i]
            
            xmat = np.vstack((xmat,self.x))
            tmat = np.vstack((tmat,t))

        return xmat, tmat
        
############# toy example ###########         
##S1 = np.array([[1,0],[-1,0],[0,1],[0,-1]])
##x1 = np.array([0,0])
##b1 = np.array([15,0.4,0.4,7,3])
##k1 = b1[0]*math.exp((-b1[1]*((0-b1[3])**2)))+b1[4]
##c1 = np.array([[k1],[0.44],[10],[0.52]])
##mat1 = np.array([[1],[x1[0]],[x1[0]],[x1[1]]])
##
##gil = Gillespie(S1,25,x1,c1,mat1,b1)
##gil.calc_gillespie()
