#==============================================================================
# returns derivative of aggregate LV (KF2) system for use in the ode solver
# inputs
# y: matrix of the elements of the system
# t: time points
# c: matrix of parameters
#==============================================================================

import numpy as np
#h11 = y[7]
#h12 = y[8]
#h21 = y[9]
#h22 = y[10]
#q11 = y[11]
#q12 = y[12]
#q22 = y[13]
def deriv_LV2(y,t,c):
        return np.array([ c[0]*y[0]-c[1]*y[0]*y[1], c[1]*y[0]*y[1]-c[2]*y[1],2*y[2]*(c[0]-c[1]*y[1])-2*y[3]*c[1]*y[0] + c[1]*y[0]*y[1]+c[0]*y[0],y[3]*(c[1]*y[0]-c[2]+c[0]-c[1]*y[1])+y[2]*c[1]*y[1]-c[1]*y[0]*y[4]-c[1]*y[0]*y[1],2*y[4]*(c[1]*y[0]-c[2])+2*y[3]*c[1]*y[1]+c[1]*y[0]*y[1]+c[2]*y[1],y[0],y[1],(c[0]-c[1]*y[1])*y[7]-c[1]*y[0]*y[8]+y[2],c[1]*y[1]*y[7]+y[8]*(c[1]*y[0]-c[2])+y[3],(c[0]-c[1]*y[1])*y[9]-y[10]*c[1]*y[0]+y[3],y[9]*c[1]*y[1]+y[10]*(c[1]*y[0]-c[2])+y[4],2*y[7],y[8]+y[9],2*y[10]])
