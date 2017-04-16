#==============================================================================
# returns gradient of non aggregate LV (KF1) for use in the ode solver
# inputs
# y: matrix of the elements of the system
# t: time points
# c: matrix of parameters
#==============================================================================

import numpy as np

def LV_jac(y,t,c):
        return np.array([[c[0]-c[1]*y[1],-c[1]*y[0],0,0,0],[c[1]*y[1],c[1]*y[0]-c[2],0,0,0],[-2*y[3]*c[1]+c[1]*y[3]+c[0],-2*c[1]*y[2]+c[1]*y[0],2*(c[0]-c[1]*y[1]),-2*c[1]*y[0],0],[c[1]*(y[3]-y[4]-y[1]),c[1]*(-y[3]+y[2]-y[0]),c[1]*y[1],c[1]*y[0]-c[2]+c[0]-c[1]*y[1],-c[1]*y[0]],[2*y[4]*c[1]+c[1]*y[1],2*y[3]*c[1]+c[1]*y[0]+c[2],0,2*c[1]*y[1],2*(c[1]*y[0]-c[2])]])
