import numpy as np

def transl_inh_fundamental_invB_integral1(t0,t1,d,b):
    
    #I = np.matrix([[2.0*b*d*(np.exp(t1)-np.exp(t0))],[0]])
    I = np.matrix([[(b/d)*(np.exp(d*t1)-np.exp(d*t0))],[0]])
    #print "I:"+str(I)
    return I