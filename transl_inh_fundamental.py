# fundamental matrix for the solution of the integrated lna in the translation inhibition example

import numpy as np

def transl_inh_fundamental(t,d):
    
    F = np.matrix([[0,0,0,0,-0.5*d*np.exp(-d*t)],[0,0,(d**2)*np.exp(-2*d*t),0,-0.5*d*np.exp(-d*t)],[0,1,0,0,0.5*np.exp(-d*t)],[0,0,-d*np.exp(-2*d*t),-0.5*d*np.exp(-d*t),(0.5-0.5*t*d)*np.exp(-d*t)],[1,0,np.exp(-2*d*t),np.exp(-d*t),t*np.exp(-d*t)]])
    
    return F