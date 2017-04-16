# inverse of the fundamental matrix for the solution of lna in the translation inhibition example

import numpy as np
#np.seterr(over='ignore')
def transl_inh_fundamental_inv1(t,d):
    F_inv = np.matrix([[np.exp(d*t),0],[-np.exp(2*d*t),np.exp(2*d*t)]])
    #print F_inv    
    return F_inv
