#==============================================================================
# returns derivative of non aggregate LV (KF1) system for use in the ode solver
# inputs
# y: matrix of the elements of the system
# t: time points
# c: matrix of parameters
#==============================================================================
import numpy as np

def deriv_LV(y,t,c):
        return np.array([ c[0]*y[0]-c[1]*y[0]*y[1], c[1]*y[0]*y[1]-c[2]*y[1],2*y[2]*(c[0]-c[1]*y[1])-2*y[3]*c[1]*y[0] + c[1]*y[0]*y[1]+c[0]*y[0],y[3]*(c[1]*y[0]-c[2]+c[0]-c[1]*y[1])+y[2]*c[1]*y[1]-c[1]*y[0]*y[4]-c[1]*y[0]*y[1],2*y[4]*(c[1]*y[0]-c[2])+2*y[3]*c[1]*y[1]+c[1]*y[0]*y[1]+c[2]*y[1]])
