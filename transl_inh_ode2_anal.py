#==============================================================================
# analytical solution of integrated lna (translation inhibition example)
# inpputs
# init : initial value
# t0 : initial time
# t1 : end time
# d : degradation rate
# b : basal rate
#==============================================================================
import numpy as np
import math
import scipy
from transl_inh_fundamental import transl_inh_fundamental
from transl_inh_fundamental_inv import transl_inh_fundamental_inv
from transl_inh_fundamental_invB_integral import transl_inh_fundamental_invB_integral

def transl_inh_ode2_anal(init,t0,t1,d,b):    
    F1 = transl_inh_fundamental(t1,d)
    F0_inv = transl_inh_fundamental_inv(t0,d)

    I = transl_inh_fundamental_invB_integral(t0,t1,d,b)
    #print "I:"+str(I)
    Y = F1*F0_inv*init +F1*I

    return Y
