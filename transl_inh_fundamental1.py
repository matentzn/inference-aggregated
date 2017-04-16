# fundamental matrix for the solution of lna in the translation inhibition example

import numpy as np

def transl_inh_fundamental1(t,d):
    
    F = np.matrix([[np.exp(-d*t),0],[np.exp(-d*t),np.exp(-2*d*t)]])
    
    return F