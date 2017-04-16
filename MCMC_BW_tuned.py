#==============================================================================
# MCMC blockwise with tuning of scale factor
# inputs
# c0: initial states
# run: total number of iterations
# likelihood: posterior
# lamda: scale factor of proposal
# outputs
# c_mat = MCMC samples
# acceptance_rate = acceptance rate
# lik_mat = matrix of posterior at each iteration
# lamda = updated scale factor
#==============================================================================

import numpy as np
import random
 
def MCMC_BW_tuned(c0,run,likelihood,lamda):
    
    dim = len(c0)
    c_mat = np.zeros([run,dim])
    c_old = c0
    accept = 0    
    likelihood0 = likelihood(c0)
    lik_mat = np.zeros([run,dim])    
    final = c0
    cov_prop = np.identity(dim)
    for i in range(run):

        new = np.random.multivariate_normal(c_old,lamda*cov_prop)

        lik_new = likelihood(new)
        ratio = lik_new-likelihood0 
        u = random.uniform(0,1)
        if np.log(u)<ratio :
            final = new
            likelihood0 = lik_new
            accept += 1
        else :
            final = c_old
        lik_mat[i,:] = likelihood0
            
        c_mat[i,:] = final
        c_old = final
        
#################### udate lamda ##############        
#        for i in range(1000,2100,100):
#            if accept < 0.2:
#                lamda = lamda*0.8
#            if accept > 0.3:
#                lamda = lamda*1.2
        if i in range(1000,5000,100):#(500,5000,100):
            #print "acceptance rate so far: "+str(float(accept)/i)
            if (float(accept)/i) < 0.1:
                lamda = lamda*0.9
            if (float(accept)/i) > 0.35:
                lamda = lamda*1.2

    #calculate acceptance rate
    acceptance_rate = np.array([float(accept)/run])
    return c_mat, acceptance_rate,lik_mat,lamda
