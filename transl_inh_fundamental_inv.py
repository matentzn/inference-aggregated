import numpy as np
#np.seterr(over='ignore')
def transl_inh_fundamental_inv(t,d):
    F_inv = np.matrix([[1.0/(d*d),1.0/(d*d),0,2.0/d,1.0],[1.0/d,0,1,0,0],[-1.*np.exp(2*d*t)/(d*d),1.0*np.exp(2*d*t)/(d*d),0,0,0],[(2.0*np.exp(d*t)*t)/(d),-2.0*np.exp(d*t)/(d**2),0,-2*np.exp(d*t)/d,0],[-2*np.exp(d*t)/d,0,0,0,0]])
    #print F_inv    
    return F_inv
