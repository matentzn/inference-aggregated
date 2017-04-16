#==============================================================================
# Ornstein-Uhlenbeck example
#likelihood 1 gives OU likelihood using KF1
#likelihood 2 gives OU likelihood using KF2
#Runs Nelder-Mead and MCMC
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from ou_simulation_1D import ou_simulation_1D
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from MCMC_BW_tuned import MCMC_BW_tuned
import random

#total iterations and burn in for MCMC
run = 50000
burn_in = 30000#45000#30000
len_chain = run-burn_in
# number of OU datasets sampled
limit = 10

####### matrices for MCMC results ##################
c1_mat1_all = np.zeros((limit,len_chain))
c2_mat1_all = np.zeros((limit,len_chain))
c1_mat2_all = np.zeros((limit,len_chain))
c2_mat2_all = np.zeros((limit,len_chain))
c1_mat1_all_log = np.zeros((limit,run))
c2_mat1_all_log = np.zeros((limit,run))
c1_mat2_all_log = np.zeros((limit,run))
c2_mat2_all_log = np.zeros((limit,run))

stat_var1_all = np.zeros((limit,len_chain))
stat_var2_all = np.zeros((limit,len_chain))

# obtaining data 
epanalipsi = 0
while epanalipsi <limit:
    print epanalipsi
    random.seed(epanalipsi)
    np.random.seed(epanalipsi)

    n = 100#200#100#50#200 #100 #50 #10 
    x0 = 20
    
    step = 1.0#2.0#1.0#0.5#2.0 #1.0 #0.5 #0.1
    print "step :  "+str(step)
    a = 4.0
    sigma = 2.0
    [obs_trapz, ou, time1, obs_anal,trace] = ou_simulation_1D(x0, n, step,a,sigma)
    real_obs = ou[1:len(ou)]
    obs = obs_anal[1:len(obs_anal)]
    #obs = real_obs
    obs_avg =  obs/step #real_obs #obs/step
    
    plt.figure(0)
    #plt.plot(obs_anal,'ko')
    plt.plot(obs,'ro')
    plt.plot(real_obs,'go')
    plt.plot(obs/step,'ko')
    #plt.xlim((0,5))
#######################################################    
    P = np.matrix([1])
    V = np.matrix([0]) 
    m0 = np.matrix([x0])
    S0 = np.matrix([0.5])
    P0 = np.matrix([1])
    V0 = np.matrix([0]) 

    #### initial updates KF using Fernhead's initialization
    
    mean0 = P0*m0.T
    cov0 = P0*S0*P0.T+V0
    y0 = np.random.multivariate_normal(np.array(mean0.T)[0], np.array(cov0))
    y0 = np.matrix(y0)
    #y0 = np.matrix([x0])
    
    lik0 = multivariate_normal.pdf(y0, mean=np.array(mean0.T)[0], cov=np.array(cov0)) #1
    llik0 = multivariate_normal.logpdf(y0, mean=np.array(mean0.T)[0], cov=np.array(cov0)) #0
    
    K_0 = S0*P0.T*(P0*S0*P0.T+V0).I
    m0_star = m0.T + K_0*(y0.T-P0*m0.T) #m0_star = m0.T
    S0_star = S0 - K_0*P0*S0 #S0_star = (S0-P0*S0)*0
        
    def likelihood1(c):
        #P = np.matrix([1/step])
        c = np.exp(c)
        c[0] = float(c[0])
        c[1] = float(c[1])
     
        init = np.array([m0_star[0,0],S0_star[0,0]])
    
        j = 0
        
        dt = 0.0001
        prodl = lik0
        suml = llik0
    
        m_star_mat = np.array([0])
        S_star_mat = np.array([0.1])
        
        for i in np.arange(0, n, step):
            
            time = np.arange(i, i+step+dt, dt)
            t0 = time[0]
            t1 = time[len(time)-1]
            delta = t1-t0
            ################################ uses analytical solutions  ##################################      
                        
            m_anal = 1.0*init[0]*np.exp(-1.0*delta*c[0])
            #print m_anal
            S_anal = ((c[1]**2)/(c[0]*2.0))*(1.0- np.exp(-2.0*delta*c[0]))+init[1]*np.exp(-2.0*delta*c[0])
            
            m = np.matrix([[m_anal]])
            S = np.matrix([[S_anal]])

            mean1 = P*m.T
            cov1 = P*S*P.T+V
            lik = multivariate_normal.pdf(obs_avg[j], mean=mean1, cov=cov1)
            llik = multivariate_normal.logpdf(obs_avg[j], mean=mean1, cov=cov1)
    
            suml += llik
            prodl *= lik
            ############################ Kalman Filter updates (KF1) ##########################
            K = S*P.T*(P*S*P.T+V).I
            m_star = m.T + K*(obs_avg[j].T-P*m.T)
            m_star_mat = np.vstack((m_star_mat, np.array(m_star).T))
            S_star = S - K*P*S
            S_star_mat = np.vstack((S_star_mat, np.array(S_star)))
            
            init = np.array([m_star[0, 0],S_star[0,0]])

            j += 1
        return suml
    
    
    def likelihood2(c):
        c = np.exp(c)
        c[0] = float(c[0])
        c[1] = float(c[1])
    
        init = np.array([m0_star[(0, 0)], S0_star[(0,0)], 0*m0_star[(0, 0)], 0, 0]) ###use with KF
                   
        j = 0
        dt = 0.0001
        prodl = lik0
        suml = llik0
    
        mx_star_mat = np.array([0])
        S_star_mat = np.array([0.1])
        
        for i in np.arange(0, n, step):
            time = np.arange(i, i+step+dt, dt)
            t0 = time[0]
            t1 = time[len(time)-1]
            delta = t1-t0
            ############### uses analytical solutions for integral of OU ############################################   
            mx_anal = 1.0*init[0]*np.exp(-1.0*delta*c[0])
            S_anal = ((c[1]**2)/(c[0]*2.0))*(1.0- np.exp(-2.0*delta*c[0]))+init[1]*np.exp(-2.0*delta*c[0])
            
            m_anal = 1.0*init[0]/c[0]*(1.0-np.exp(-1.0*delta*c[0]))        
            #QM_anal = c[1]*(c[0]**2)*0.5+(-c[1]*(c[0]**2)+init[1]*c[0]+init[1])*np.exp(-(time[len(time)-1]-time[0])/(c[0]))+(c[1]*(c[0]**2)*0.5-init[1]*c[0])*np.exp(-2*(time[len(time)-1]-time[0])/c[0])
            #QM_anal = 0.5*((c[1]**2.0)/(c[0]**2))+(-(c[1]**2.0)/(c[0]**2.0)+1.0*init[1]/c[0])*np.exp(-1.0*delta*(c[0]))+(0.5*((c[1]**2.0)/(c[0]**2.0))-1.0*init[1]/c[0])*np.exp(-2.0*delta*c[0])
            QM_anal = 0.5*((c[1]**2.0)/(c[0]**2.0))+(-(c[1]**2.0)/(c[0]**2.0)+1.0*init[1]/c[0])*np.exp(-1.0*delta*(c[0]))+(0.5*((c[1]**2.0)/(c[0]**2.0))-1.0*init[1]/c[0])*np.exp(-2.0*delta*c[0])
    
            #Q_anal = c[1]*(c[0]**2)*(time[len(time)-1]-time[0])+init[1]-2*(c[1]*c[0]**3-init[1]*c[0]**2-init[1]*c[0])*(1- np.exp(-(time[len(time)-1]-time[0])/(c[0])))+(0.5*c[1]*c[0]**3-init[1]*c[0]**2)*(1- np.exp(-2*(time[len(time)-1]-time[0])/(c[0])))
            #Q_anal = ((c[1]**2.0)/((float(c[0])**2)))*delta*1.0-2.0*((c[1]**2.0)/c[0]**3.0-1.0*init[1]/(c[0]**2.0))*(1.0- np.exp(-delta*c[0]*1.0))+(0.5*(c[1]**2.0)/(c[0]**3.0)-init[1]*1.0/(c[0]**2.0))*(1.0- np.exp(-2.0*delta*(c[0])))
            Q_anal = 1.0*((c[1]**2.0)/((float(c[0])**2)))*delta-2.0*((c[1]**2.0)/c[0]**3.0-1.0*init[1]/(c[0]**2.0))*(1.0- np.exp(-1.0*delta*c[0]))+(0.5*(c[1]**2.0)/(c[0]**3.0)-1.0*init[1]/(c[0]**2.0))*(1.0- np.exp(-2.0*delta*(c[0])))
    
            mx = np.matrix([[mx_anal]])
            S = np.matrix([[S_anal]])
            m = np.matrix([[m_anal]])
            QM = np.matrix([[QM_anal]])
            Q = np.matrix([[Q_anal]])
                
            mean1 = P*m.T
            cov1 = P*Q*P.T+V
            lik = multivariate_normal.pdf(obs[j], mean=mean1, cov=cov1)
            llik = multivariate_normal.logpdf(obs[j], mean=mean1, cov=cov1)
            
            suml += llik
            prodl *= lik
                        
    ###################### Kalman Filter updates (KF2) #######################
            part1x = QM.T*P.T*(P*Q*P.T+V).I
            mx_star = mx.T + part1x*(obs[j].T-P*m.T)
            mx_star_mat = np.vstack((mx_star_mat, np.array(mx_star).T))
            
            S_star = S - part1x*P*QM
            S_star_mat = np.vstack((S_star_mat, np.array(S_star)))
            #print mx_star, S_star
            init = np.array([mx_star[(0, 0)], S_star[(0, 0)], 0*mx_star[(0, 0)], 0, 0])

            j +=1
        return suml
    
    def neg_likelihood2(c):
        return -likelihood2(c)
    
    def neg_likelihood1(c):
        return -likelihood1(c)
    
    def prior(c):
        c = np.exp(c)
        for i in range(len(c)):
            if c[i] <0 :
                return -np.inf
        return 0
        
    def posterior1(c):
       return likelihood1(c)+prior(c)
       
    def posterior2(c):
       return likelihood2(c)+prior(c)
       
    c0 = np.array([0.1,0.1])
    c0 = np.log(c0)
    optim1 = minimize(neg_likelihood1,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
    print np.exp(optim1.x)
    
    
    optim2 = minimize(neg_likelihood2,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
    print np.exp(optim2.x)
    
    c_real = np.array([4,2])
    lik1_real = likelihood1(c_real)
    print 'Likelihood1-c_real'
    print 'Likelihood1-c_real',lik1_real

    ############## run MCMC likelihood1 #########################################################################
    n_chains = 1
    #len_chain = run-burn_in
    chain_param1 = np.zeros([n_chains,len_chain])
    chain_param2 = np.zeros([n_chains,len_chain])
    chain_param12 = np.zeros([n_chains,len_chain])
    chain_param22 = np.zeros([n_chains,len_chain])
    for ichain in range(0,n_chains):
        
        lamda1 = 0.1
        c0 = np.array([random.uniform(0,10),random.uniform(0,10)])
        c0 = np.log(c0)
    
        print c0
        
        [c1_mat,rate1,log_lik1,lamda_1] = MCMC_BW_tuned(c0,run,posterior1,lamda1)
        c1_mat1_all_log[epanalipsi,:] = c1_mat[:,0]
        c2_mat1_all_log[epanalipsi,:] = c1_mat[:,1]
        #c1_mat1_all_log.dump("c1_mat1_all_log_step2.dat")
        #c2_mat1_all_log.dump("c2_mat1_all_log_step2.dat")
        
        c1_mat = np.exp(c1_mat)
        
        c1_mat1 = c1_mat[:,0]
        c2_mat1 = c1_mat[:,1]
                
        thinned11 = []
        thinned21 = []

        lik_thinned1 = []
        stationary_var1 = []
        for kept_samples in range(burn_in,run):
            #if (n % 2 == 0):
            thinned11.append(c1_mat1[kept_samples])
            thinned21.append(c2_mat1[kept_samples])
            lik_thinned1.append(log_lik1[kept_samples])
            stationary_var1.append(0.5*((c2_mat1[kept_samples])**2)/c1_mat1[kept_samples])
                
        print "Mean1:  "+str(np.mean(thinned11))
        print "Sigma1: "+str(np.std(thinned11))
        print "Mean2:  "+str(np.mean(thinned21))
        print "Sigma2: "+str(np.std(thinned21))
        print "Mean3:  "+str(np.mean(stationary_var1))
        print "Sigma3: "+str(np.std(stationary_var1))
        
        c1_mat1_all[epanalipsi,:] = thinned11
        c2_mat1_all[epanalipsi,:] = thinned21
        stat_var1_all[epanalipsi,:] = stationary_var1
        
        length = range(run)
        plt.figure(3)
        plt.plot(c1_mat1)
        plt.ylim((0,1.0))
        
        mean_c11 = np.zeros(len(length))
        
        for k in range(run) :
            mean_c11[k] = np.mean(c1_mat1[0:k+1])
        plt.figure(4)
        plt.plot(length,mean_c11)
        plt.figure(5)
        plt.plot(c2_mat1)
        plt.ylim((0,1.0))
                
        mean_c21 = np.zeros(len(length))
        
        for k in range(run) :
            mean_c21[k] = np.mean(c2_mat1[0:k+1])
        plt.figure(6)
        plt.plot(length,mean_c21)
        
        plt.figure(12)
        plt.hist(stationary_var1, 5, histtype='step')
        plt.vlines(0.5,0,1000.0,colors=u'r')
        plt.xlabel(r'$v_1$', fontsize=24)
        
        drift1 = thinned11
        diffus1 = thinned21
        ###### keep the different chains #####
        chain_param1[ichain] = thinned11
        chain_param2[ichain] = thinned21
        
    var_of_each_chain1 = np.zeros(n_chains)
    mean_of_each_chain1 = np.zeros(n_chains)
    
    for i in range(0,n_chains):
        var_of_each_chain1[i] = np.var(chain_param1[i],ddof=1)
        mean_of_each_chain1[i] = np.mean(chain_param1[i])
    
    W1 = np.mean(var_of_each_chain1) 
    #mean_of_mean = np.mean(mean_of_each_chain)
    B1 = len_chain*(np.var(mean_of_each_chain1))#,ddof=1))  #variance of the chain means * (samples in each chain)
    V_hat1 = (1-1.0/len_chain)*W1+(1.0/len_chain)*B1
    R_hat1 = np.sqrt(V_hat1/W1)
    
        
    var_of_each_chain2 = np.zeros(n_chains)
    mean_of_each_chain2 = np.zeros(n_chains)
    
    for i in range(0,n_chains):
        var_of_each_chain2[i] = np.var(chain_param2[i])#,ddof=1)
        mean_of_each_chain2[i] = np.mean(chain_param2[i])
    
    W2 = np.mean(var_of_each_chain2) 
    #mean_of_mean = np.mean(mean_of_each_chain)
    B2 = len_chain*(np.var(mean_of_each_chain2))#,ddof=1))  #variance of the chain means * (samples in each chain)
    V_hat2 = (1-1.0/len_chain)*W2+(1.0/len_chain)*B2
    R_hat2 = np.sqrt(V_hat2/W2)
    
      
        
    
    ######### run MCMC likelihood2 ########################################################
    
    for ichain in range(0,n_chains):
        
        lamda2 = 0.1
        c0 = np.array([random.uniform(0,10),random.uniform(0,10)])
        c0 = np.log(c0)
        print c0
        
        [c2_mat,rate2,log_lik2,lamda_2] = MCMC_BW_tuned(c0,run,posterior2,lamda2)
        
        c1_mat2_all_log[epanalipsi,:] = c2_mat[:,0]
        c2_mat2_all_log[epanalipsi,:] = c2_mat[:,1]   
        #c1_mat2_all_log.dump("c1_mat2_all_log_step2.dat")
        #c2_mat2_all_log.dump("c2_mat2_all_log_step2.dat")
        
         
        c2_mat = np.exp(c2_mat)
        c1_mat2 = c2_mat[:,0]
        c2_mat2 = c2_mat[:,1]
                
        thinned1 = []
        thinned2 = []
        lik_thinned2 = []
        stationary_var2 = []
        for kept_samples in range(burn_in,run):
            #if (n % 2 == 0):
            thinned1.append(c1_mat2[kept_samples])
            thinned2.append(c2_mat2[kept_samples])
            lik_thinned1.append(log_lik2[kept_samples])
            stationary_var2.append(0.5*((c2_mat2[kept_samples])**2)/c1_mat2[kept_samples])
                
        print "Mean1:  "+str(np.mean(thinned1))
        print "Sigma1: "+str(np.std(thinned1))
        print "Mean2:  "+str(np.mean(thinned2))
        print "Sigma2: "+str(np.std(thinned2))
        print "Mean3:  "+str(np.mean(stationary_var2))
        print "Sigma3: "+str(np.std(stationary_var2))
        #c1_mat2_all[epanalipsi,:] = thinned1
        #c2_mat2_all[epanalipsi,:] = thinned2
        stat_var2_all[epanalipsi,:] = stationary_var2

        length = range(run)
        plt.figure(23)
        plt.plot(c1_mat2)
        
        mean_c12 = np.zeros(len(length))
        
        for k in range(run) :
            mean_c12[k] = np.mean(c1_mat2[0:k+1])
        plt.figure(24)
        plt.plot(length,mean_c12)
        plt.figure(25)
        plt.plot(c2_mat2)
        
        mean_c22 = np.zeros(len(length))
        
        for k in range(run) :
            mean_c22[k] = np.mean(c2_mat2[0:k+1])
        plt.figure(26)
        plt.plot(length,mean_c22)
        
        plt.figure(22)
        plt.hist(stationary_var2, 5, histtype='step')
        plt.vlines(0.5,0,1000.0,colors=u'r')
        plt.xlabel(r'$v_1$', fontsize=24)
        
        ###### keep the different chains #####
        chain_param12[ichain] = thinned1
        chain_param22[ichain] = thinned2
        
    var_of_each_chain21 = np.zeros(n_chains)
    mean_of_each_chain21 = np.zeros(n_chains)
    
    for i in range(0,n_chains):
        var_of_each_chain21[i] = np.var(chain_param12[i])#,ddof=1)
        mean_of_each_chain21[i] = np.mean(chain_param12[i])
    
    W1 = np.mean(var_of_each_chain2) 
    #mean_of_mean = np.mean(mean_of_each_chain)
    B1 = len_chain*(np.var(mean_of_each_chain21))#,ddof=1))  #variance of the chain means * (samples in each chain)
    V_hat1 = (1-1.0/len_chain)*W1+(1.0/len_chain)*B1
    R_hat12 = np.sqrt(V_hat1/W1)
    
        
    var_of_each_chain22 = np.zeros(n_chains)
    mean_of_each_chain22 = np.zeros(n_chains)
    
    for i in range(0,n_chains):
        var_of_each_chain22[i] = np.var(chain_param22[i])##,ddof=1)
        mean_of_each_chain22[i] = np.mean(chain_param22[i])
    
    W2 = np.mean(var_of_each_chain22) 
    #mean_of_mean = np.mean(mean_of_each_chain)
    B2 = len_chain*(np.var(mean_of_each_chain22))#,ddof=1))  #variance of the chain means * (samples in each chain)
    V_hat2 = (1-1.0/len_chain)*W2+(1.0/len_chain)*B2
    R_hat22 = np.sqrt(V_hat2/W2)
 
    plt.figure(21)
    plt.hist(thinned1, 5,normed=True, histtype='step',color = 'g')
    plt.hist(drift1, 5,normed=True, histtype='step',color = 'b')
    plt.vlines(c_real[0],0,2.0,colors=u'r')
    plt.xlabel(r'$\tau$', fontsize=24)
    plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
    plt.show()
    
    plt.figure(22)
    plt.hist(thinned2, 5,normed=True, histtype='step',color = 'g')
    plt.hist(diffus1, 5,normed=True, histtype='step',color = 'b')
    plt.vlines(c_real[1],0,2.0,colors=u'r')
    plt.xlabel(r'$c$', fontsize=24)
    plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
    plt.show()
    
    epanalipsi += 1
################## save the results #######################
#c1_mat1_all_log.dump("c1_mat1_all01.dat")
#c2_mat1_all_log.dump("c2_mat1_all01.dat")
#stat_var1_all.dump("stat_var1_all01.dat")
#
#c1_mat2_all_log.dump("c1_mat2_all01.dat")
#c2_mat2_all_log.dump("c2_mat2_all01.dat")
#stat_var2_all.dump("stat_var2_all01.dat")
#
#import pickle
#with open("c1_mat1_all_step01.bin", "wb") as output1:
#    pickle.dump(c1_mat1_all_log, output1)
#    
#with open("c2_mat1_all_step01.bin", "wb") as output2:
#    pickle.dump(c2_mat1_all_log, output2)    
#with open("stat_var1_step01.bin", "wb") as output3:
#    pickle.dump(stat_var1_all, output3)
#
#with open("c1_mat2_all_step01.bin", "wb") as output4:
#    pickle.dump(c1_mat2_all_log, output4)
#    
#with open("c2_mat2_all_step01.bin", "wb") as output5:
#    pickle.dump(c2_mat2_all_log, output5)    
#with open("stat_var2_step01.bin", "wb") as output6:
#    pickle.dump(stat_var2_all, output6)
    
#plt.boxplot([stat_var1_all[0]],whis="range")
#

c1_1_all_exp = np.exp(c1_mat1_all_log)
c2_1_all_exp = np.exp(c2_mat1_all_log)
c1_2_all_exp = np.exp(c1_mat2_all_log)
c2_2_all_exp = np.exp(c2_mat2_all_log)


mean1_1 = np.mean(c1_1_all_exp[:,burn_in:run])
std1_1 = np.std(c1_1_all_exp[:,burn_in:run])
mean2_1 = np.mean(c2_1_all_exp[:,burn_in:run])
std2_1 = np.std(c2_1_all_exp[:,burn_in:run])
print('KF1:')
print np.around(mean1_1,3)
print np.around(std1_1,3)
print np.around(mean2_1,3)
print np.around(std2_1,3)

mean1_2 = np.mean(c1_2_all_exp[:,burn_in:run])
std1_2 = np.std(c1_2_all_exp[:,burn_in:run])
mean2_2 = np.mean(c2_2_all_exp[:,burn_in:run])
std2_2 = np.std(c2_2_all_exp[:,burn_in:run])
print('KF2:')
print np.around(mean1_2,3)
print np.around(std1_2,3)
print np.around(mean2_2,3)
print np.around(std2_2,3)

