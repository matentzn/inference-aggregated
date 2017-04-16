#==============================================================================
# analytical solution of lna (translation inhibition example)
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
from transl_inh_fundamental1 import transl_inh_fundamental1
from transl_inh_fundamental_inv1 import transl_inh_fundamental_inv1
from transl_inh_fundamental_invB_integral1 import transl_inh_fundamental_invB_integral1

def transl_inh_ode_anal(init,t0,t1,d,b):    
    F1 = transl_inh_fundamental1(t1,d)
    F0_inv = transl_inh_fundamental_inv1(t0,d)

    I = transl_inh_fundamental_invB_integral1(t0,t1,d,b)
    #print "I:"+str(I)
    Y = F1*F0_inv*init +F1*I

    return Y
