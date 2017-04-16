#==============================================================================
# adaptive MCMC 
# inputs
# c0: initial states
# run: total number of iterations
# likelihood: posterior
# lamda: scale factor of proposal
# outputs
# c_mat = MCMC samples
# acceptance_rate = acceptance rate
# lik_mat = matrix of posterior at each iteration
# lamda_adapt = scale factor of adaptive kernel
#==============================================================================

import numpy as np
import random

def MCMC_BW_adapted_roberts(c0,run,likelihood,lamda):
    dim = len(c0)
    c_mat = np.zeros([run,dim])
    c_old = c0
    accept = 0   
    likelihood0 = likelihood(c0)
    lik_mat = np.zeros([run,dim])
    final = c0
    cov_init = np.identity(dim)
    cov_prop = cov_init
    lamda_adapt = 1./np.sqrt(dim)
    i_adapt = 0
    start_adapt = 5000
    for i in range(run):
       
        if i <=start_adapt:
            new = np.random.multivariate_normal(c_old,lamda*cov_init)
        else:
            u_adapt = random.uniform(0,1)
            if u_adapt < 0.95:
                #print "check lamda: "+str(lamda_adapt)
                #print "check covar: "+str(cov_prop)
                #print "check lamda*covar: "+str(lamda_adapt*cov_prop)
                new = np.random.multivariate_normal(c_old,(lamda_adapt**2)*cov_prop)
                #print "check new: "+str(new)
            else :
                new = np.random.multivariate_normal(c_old,lamda*cov_init)
               
        lik_new = likelihood(new)
        ratio = lik_new-likelihood0
        u = random.uniform(0,1)
        if np.log(u)<ratio :
            final = new
            likelihood0 = lik_new
            accept += 1
            if i>start_adapt:
                i_adapt += 1                
                if u_adapt < 0.95:
                    lamda_adapt = lamda_adapt + ((2.3*0.1)/np.sqrt(i_adapt))
               
        else :
            final = c_old
            if i>start_adapt:
                i_adapt += 1                
                if u_adapt < 0.95:
                    lamda_adapt = lamda_adapt - (0.1/np.sqrt(i_adapt))
               
              
       
        lik_mat[i,:] = likelihood0
        c_mat[i,:] = final
        c_old = final
       
        if i in range(1000,start_adapt,100):#(500,5000,100):
            print "acceptance rate so far: "+str(float(accept)/i)
            if (float(accept)/i) < 0.1:
                lamda = lamda*0.9
            if (float(accept)/i) > 0.35:
                lamda = lamda*1.2
       
        if i>=start_adapt:#(1000,1500,2000):
            cov_all = c_mat[(start_adapt-1000):i,0]

            for j in range(1,dim):
                c_mat_dim = c_mat[(start_adapt-1000):i,j]
                cov_all = np.vstack([cov_all,c_mat_dim])
            cov_prop = np.cov(cov_all)
   #calculate acceptance rate
    print 'lamda:'+str(lamda)
    acceptance_rate = np.array([float(accept)/run])
    return c_mat, acceptance_rate,lik_mat,lamda_adapt#step_size