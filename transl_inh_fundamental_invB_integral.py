# inverse of fundamental matrix for the solution of the integrated lna in the translation inhibition example

import numpy as np

def transl_inh_fundamental_invB_integral(t0,t1,d,b):
    
    I = np.matrix([[2.0*b*(t1-t0)/(d*d)],[1.0*b*(t1-t0)/(d)],[0],[(-2*b/(d*d*d))*(np.exp(d*t0)*(-2+d*t0)+np.exp(d*t1)*(2-d*t1))],[(-2*b/d**2)*(np.exp(d*t1)-np.exp(d*t0))]])
    #print "I:"+str(I)
    return I