#==============================================================================
#Translation inhibition example with synthetic data
#likelihood 1 gives likelihood using KF1
#likelihood 2 gives likelihood using KF2
#Runs adaptive MCMC 
#==============================================================================
import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
#from gil_integr_transl_inh import gil_integr_transl_inh
#from lna_simulation_protein import lna_simulation_protein
#from scipy.optimize import minimize
from scipy.stats import multivariate_normal
#from trans_inh_ode import trans_inh_ode
#from trans_inh_ode2 import trans_inh_ode2
from transl_inh_ode2_anal import transl_inh_ode2_anal
from transl_inh_ode_anal import transl_inh_ode_anal
from MCMC_BW_adapted_roberts import MCMC_BW_adapted_roberts


#from scipy.stats import gamma
from scipy.stats import kde
#from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import uniform
#import emcee
#import triangle 
import random 
import pickle 

random.seed(50)
np.random.seed(50)

m0_1 = 400#200 #500 #100 #2000 
n = 9 #20
length_of_interval = n
aggr_step = 0.5

cp = 200#60#160#60# 6.5#5.7 #3.5
dp = 0.97#0.8#1.0#0.8#0.9 #0.8
k_const = 0.03#0.7#0.03#0.7 #1.0
noise_const = 0.1#1.0#1e-6 #1.0

################## c = [cp,dp,noise,k,c01,...,c0n]
num_rep = 30#30#40#37 #use even if splitting

###################### import synthetic dataset ################################
with open("obs_transl_seed50.bin", "rb") as data1:
   obs = pickle.load(data1)

with open("obs_avg_transl_seed50.bin", "rb") as data2:
   obs_avg = pickle.load(data2)

######################### create the dataset #######################################
#real = np.zeros((num_rep,(n/aggr_step)))
#obs = np.zeros((num_rep,(n/aggr_step)))
#obs_avg = np.zeros((num_rep,(n/aggr_step)))
#
#for rep in range(0,num_rep):
#    ##[obs_trapz,real_obs, obs_integr,time1, species] = lna_simulation_protein(m0_1,n,aggr_step,cp,dp)
#    [obs_trapz,real_obs, obs_integr,time1, species] = gil_integr_transl_inh(m0_1,n,aggr_step,cp,dp)
#    obs1 = obs_integr[:,0]*k_const
#    #print len(obs1)
#    real_obs1 = real_obs[:,0]
#    noise = np.matrix(np.random.normal(0,noise_const,len(obs1))).T
#    obs_avg1 =  obs1/aggr_step  #real_obs1*k_const
#    obs1 = obs1+noise
#    obs_avg1 = obs_avg1+noise#real_obs1+noise
#    real[rep,:] = real_obs1.T
#    obs[rep,:] = obs1.T
#    obs_avg[rep,:] = obs_avg1.T
#
#xronos = np.arange(aggr_step,n+aggr_step,aggr_step)
#plt.figure(1)
#plt.plot(xronos,obs_trapz[:,0],'co')
#plt.plot(xronos,obs_integr[:,0]/aggr_step,'mo')
#plt.plot(xronos,real_obs[:,0],'go')
#plt.show()
################################################################################################
c_real = np.array([cp,dp,noise_const,k_const,m0_1]) 

print "ground truth:  "+str(c_real)

def likelihood1(c) :
    c = np.exp(c)
    c[0] = c[0]/c[3]
    c[4] = c[4]/c[3]
    
    V = np.matrix([c[2]**2])
    #V = np.matrix([c[2]])
    
    S0 = np.matrix([0.001])
    #V0 = np.matrix([0.0])
    P0 = np.matrix([1.0])
    P = np.matrix([float(c[3])])  
    m0 = np.matrix([c[4]])
    #P = np.matrix([1.0])    
    sum_all = 0
    for r in range(0,num_rep):

        lik0 = 1
        llik0 = 0

        m0_star = m0.T 
        S0_star = (S0-P0*S0)*0
        #init = np.array([m0_star[(0,0)],S0_star[(0,0)]]) #to be used for numerical solver
        init = np.matrix([[m0_star[(0,0)]],[S0_star[(0,0)]]])
 
        j = 0
        step = aggr_step
        #dt = 0.001 # for numerical solver
        prodl = lik0
        suml = llik0
        m_star_mat = np.array([0])
        S_star_mat = np.array([1])
        for i in np.arange(0, n, step):
######################################## numerical solution #######################################            
#            time = np.arange(i, i+step+dt, dt)
#            #y = odeint(sge2_3,init,time,args = (c,b,))
#            y = odeint(trans_inh_ode,init,time,args = (c,))        
#            l = len(y)
#                  
#            m = np.matrix([y[l-1,0]])
#            S = np.matrix([y[l-1,1]])    
############################ analytical solution ######################################################      
            t0 = i
            #print t0
            t1 = i+step
            #print t1
            #time = np.arange(i, i+step+dt, dt)
            #y = odeint(trans_inh_ode2,init,time,args = (c,)) 
            #l = len(y)
            y = transl_inh_ode_anal(init,t0,t1,c[1],c[0])
                  
            S = np.matrix([y[1,0]])    
            m = np.matrix([y[0,0]])

###########################################################################################################            
            mean1 = P*m.T
            cov1 = P*S*P.T+V
            try:
                multivariate_normal.pdf(obs[r,j], mean=mean1, cov=cov1)
            except ValueError:
                print "c1:  "+str(c)
                print "value1:  "+str(obs[r,j])
                print "j1:  "+str(j)
                print "mean1:  "+str(mean1)
                print "cov1:  "+str(cov1)
                print "init1:  "+str(init)
            lik = multivariate_normal.pdf(obs_avg[r,j], mean=mean1, cov=cov1)
            llik = multivariate_normal.logpdf(obs_avg[r,j], mean=mean1, cov=cov1)
            suml += llik
            prodl *= lik
            
            part1 = S.T*P.T*(P*S*P.T+V).I
            m_star = m.T + part1*(obs_avg[r,j].T-P*m.T)
    
            for spec in range(0,1):
                    if m_star[(spec,0)]<0:
                        #print 'negative mx2'
                        m_star[(spec,0)] = 0
            S_star = S - part1*P*S
            m_star_mat = np.vstack((m_star_mat, np.array(m_star).T))
            S_star_mat = np.vstack((S_star_mat, np.array(S_star)))
            #init = np.array([m_star[(0,0)],S_star[(0,0)]])
            init = np.matrix([[m_star[(0,0)]],[S_star[(0,0)]]])
    
            j +=1
        sum_all = sum_all + suml
    return sum_all


def likelihood2(c) :
    c = np.exp(c)
    
    c[0] = c[0]/c[3]
    c[4] = c[4]/c[3]
    
    V = np.matrix([c[2]**2])
    #V = np.matrix([c[2]])

    S0 = np.matrix([0.001])
    #V0 = np.matrix([0.0])
    #P0 = np.matrix([float(c[2])])
    P0 = np.matrix([1.0])
    P = np.matrix([float(c[3])])  
    m0 = np.matrix([c[4]])
    #P = np.matrix([1.0])    
    sum_all = 0
    for r in range(0,num_rep):
    
        lik0 = 1
        llik0 = 0

        m0_star = m0.T 
        S0_star = (S0-P0*S0)*0
        init = np.matrix([[m0_star[(0,0)]],[S0_star[(0,0)]],[0],[0],[0]])    
        #init = np.array([m0_star[(0,0)],S0_star[(0,0)],0,0,0]) #to be used for numerical solver
 
        j = 0
        step = aggr_step
        #dt = 0.001 #for numerical solver
        prodl = lik0
        suml = llik0
        mx_star_mat = np.array([0])
        S_star_mat = np.array([1])
        for i in np.arange(0, n, step):
########################################### numerical solution ####################################            
#            time = np.arange(i, i+step+dt, dt)
#            #y = odeint(sge2_3,init,time,args = (c,b,))
#            y = odeint(trans_inh_ode2,init,time,args = (c,))        
#            l = len(y)
#                  
#            mx = np.matrix([y[l-1,0]])
#            S = np.matrix([y[l-1,1]])    
#            m = np.matrix([y[l-1,2]])
#            Q = np.matrix([y[l-1,4]])
#            QM = np.matrix([y[l-1,3]])
############################################ analytical solution ########################
            t0 = i
            #print t0
            t1 = i+step
            #print t1
            #time = np.arange(i, i+step+dt, dt)
            #y = odeint(trans_inh_ode2,init,time,args = (c,)) 
            #l = len(y)
            y = transl_inh_ode2_anal(init,t0,t1,c[1],c[0])
                  
            mx = np.matrix([y[0,0]])
            S = np.matrix([y[1,0]])    
            m = np.matrix([y[2,0]])
            Q = np.matrix([y[4,0]])
            QM = np.matrix([y[3,0]])
##################################################################################            
            mean1 = P*m.T
            cov1 = P*Q*P.T+V
            try:
                multivariate_normal.pdf(obs[r,j], mean=mean1, cov=cov1)
            except ValueError:
                print "c2:  "+str(c)
                print "value2:  "+str(obs[r,j])
                print "j2:  "+str(j)
                print "mean2:  "+str(mean1)
                print "cov2:  "+str(cov1)
                print "init2:  "+str(init)
            lik = multivariate_normal.pdf(obs[r,j], mean=mean1, cov=cov1)
            llik = multivariate_normal.logpdf(obs[r,j], mean=mean1, cov=cov1)
            suml += llik
            prodl *= lik
            
            part1x = QM.T*P.T*(P*Q*P.T+V).I
            mx_star = mx.T + part1x*(obs[r,j].T-P*m.T)
    
            for spec in range(0,1):
                    if mx_star[(spec,0)]<0:
                        #print 'negative mx2'
                        mx_star[(spec,0)] = 0
            S_star = S - part1x*P*QM
            mx_star_mat = np.vstack((mx_star_mat, np.array(mx_star).T))
            S_star_mat = np.vstack((S_star_mat, np.array(S_star)))
            init = np.matrix([[mx_star[(0,0)]],[S_star[(0,0)]],[0],[0],[0]])
            #init = np.array([mx_star[(0,0)],S_star[(0,0)],0,0,0])
    
            j +=1
        sum_all = sum_all + suml
    return sum_all

m0_all_trans = m0_1*k_const*np.ones((2,1)).T
c_rest_trans = np.array([cp*k_const,dp,noise_const**2,k_const])
c_real_trans = np.hstack((c_rest_trans,m0_all_trans[0]))
#print 'Likelihood2 at real values:' +str(likelihood2(np.log(c_real_trans)))

#================================== Nelder Mead ============================================
# def neg_likelihood1(c):
#     return -likelihood1(c)
# 
# 
# def neg_likelihood2(c):
#     return -likelihood2(c)
#     
# c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])
# #c0 = np.array([expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0)])
# 
# c0 = np.log(c0)
# 
# print c0
# optim1 = minimize(neg_likelihood1,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
# print optim1
# exp_optim1 = np.exp(optim1.x)
# exp_optim1[0] = exp_optim1[0]/exp_optim1[3]
# for i in range(4,5):
#     exp_optim1[i] = exp_optim1[i]/exp_optim1[3]
# print exp_optim1
# 
# print c0
# 
# optim2 = minimize(neg_likelihood2,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
# print optim2
# exp_optim2 = np.exp(optim2.x)
# exp_optim2[0] = exp_optim2[0]/exp_optim2[3]
# for i in range(4,5):
#     exp_optim2[i] = exp_optim2[i]/exp_optim2[3]
# print exp_optim2
#==============================================================================
def posterior1(c):
    exp_c = np.exp(c)
#    sum_prior_m0 = 0
#    for i in range(0,1):
#        sum_prior_m0 += norm.logpdf(exp_c[i+4]/exp_c[3],m0_1,1.)
        #sum_prior_m0 += norm.logpdf(exp_c[i+4],m0_1,1.)
        
    #return likelihood1(c)+ c[0]+c[1]+c[2]-c[3]+c[4]# #+sum_prior_m0
    #return likelihood1(c) + norm.logpdf(c[4],np.log(m0_1*k_const),1.)    
    return likelihood1(c)+ expon.logpdf(float(exp_c[0]/exp_c[3]),loc=0, scale=10000) +expon.logpdf(float(exp_c[1]),loc=0, scale=10000)+expon.logpdf(float(exp_c[2]),loc=0, scale=10000)+expon.logpdf(float(exp_c[3]),loc=0, scale=10000)+expon.logpdf(float(exp_c[4]/exp_c[3]),loc=0, scale=10000) +c[0]+c[1]+c[2]-c[3]+c[4]

def posterior2(c):
    exp_c = np.exp(c)
#    sum_prior_m0 = 0
#    for i in range(0,1):
#        sum_prior_m0 += norm.logpdf(exp_c[i+4]/exp_c[3],m0_1,1.)
        
    #return likelihood2(c)+ c[0]+c[1]+c[2]-c[3]+c[4]#+sum_prior_m0
    #return likelihood2(c) + norm.logpdf(c[4],np.log(m0_1*k_const),1.)    
    return likelihood2(c)+ expon.logpdf(float(exp_c[0]/exp_c[3]),loc=0, scale=10000) +expon.logpdf(float(exp_c[1]),loc=0, scale=10000)+expon.logpdf(float(exp_c[2]),loc=0, scale=10000)+expon.logpdf(float(exp_c[3]),loc=0, scale=10000)+expon.logpdf(float(exp_c[4]/exp_c[3]),loc=0, scale=10000) +c[0]+c[1]+c[2]-c[3]+c[4]
        
#print 'Posterior2 at real values:' +str(posterior2(np.log(c_real_trans)))

###################################################### MCMC2 #########################################
run = 30000#50000
burn_in = 10000#30000
lamda1 = 0.1
n_chains = 1
len_chain = run-burn_in
lamda2 = 0.1
#n_chains = 2
len_chain = run-burn_in
########################################### KF1 #####################################################
chain_param1 = np.zeros([n_chains,len_chain])
chain_param2 = np.zeros([n_chains,len_chain])
chain_param3 = np.zeros([n_chains,len_chain])
chain_param4 = np.zeros([n_chains,len_chain])
chain_param5 = np.zeros([n_chains,len_chain])
chain_param1_all_1 = np.zeros([n_chains,run])
chain_param2_all_1 = np.zeros([n_chains,run])
chain_param3_all_1 = np.zeros([n_chains,run])
chain_param4_all_1 = np.zeros([n_chains,run])
chain_param5_all_1 = np.zeros([n_chains,run])

chain_lik2 = np.zeros([n_chains,run])
for ichain in range(0,n_chains):
    #c0_minus = np.array([expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0)])    
    #c0_minus = np.array([cp*k_const,dp,noise_const**2,k_const])    
    c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])
    print c0
    c0 = np.log(c0)
    while True:
        try:
            [c1_mat,rate1,lik1_mat,lambda1] = MCMC_BW_adapted_roberts(c0,run,posterior1,lamda1)
        except ValueError:
            print 'Try new c0'
            c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])
            c0 = np.log(c0)
            print c0
            #[c1_mat,rate1,lik1,lamda_1] = MCMC_BW_adapted_roberts(c0,run,posterior1,lamda1)
            continue
        break
    print "Rate1:  "+str(rate1)
    c1_mat = np.exp(c1_mat)
    #c1_mat.dump("c1_exp_mat_transl_inh_seed50_expprior_new_roberts3b.dat")
    
    c12_mat = c1_mat[:,0]/c1_mat[:,3]
    c22_mat = c1_mat[:,1]
    c32_mat = c1_mat[:,2]
    c42_mat = c1_mat[:,3]
    c52_mat = c1_mat[:,4]/c1_mat[:,3]

    lik12_mat = lik1_mat[:,0]
    
    plt.figure(157)
    plt.plot(lik12_mat)
    #plt.ylim([-525,-535])
    
    plt.figure(37)
    plt.plot(c12_mat)
    #plt.ylim((75,85))
    plt.figure(38)
    plt.plot(c22_mat)
    plt.figure(39)
    plt.plot(c32_mat)
    #plt.ylim((4.6,5.3))
    plt.figure(40)
    plt.plot(c42_mat)
    #plt.ylim((0.075,0.078))
    plt.figure(41)
    plt.plot(c52_mat)
    
    thinned21 = []
    thinned22 = []
    thinned23 = []
    thinned24 = []
    thinned25 = []
    thinned26 = []
    
    for kept in range(burn_in,run):
        #if (n % 2 == 0):
        thinned21.append(c12_mat[kept])
        thinned22.append(c22_mat[kept])
        thinned23.append(c32_mat[kept])
        thinned24.append(c42_mat[kept])
        thinned25.append(c52_mat[kept])
        
    print "Mean1:  "+str(np.mean(thinned21))
    print "Sigma1: "+str(np.std(thinned21))
    print "Mean2:  "+str(np.mean(thinned22))
    print "Sigma2: "+str(np.std(thinned22))
    print "Mean3:  "+str(np.mean(thinned23))
    print "Sigma3: "+str(np.std(thinned23))
    print "Mean4:  "+str(np.mean(thinned24))
    print "Sigma4: "+str(np.std(thinned24))
    print "Mean5:  "+str(np.mean(thinned25))
    print "Sigma5: "+str(np.std(thinned25))
    
    
    length = range(run)
    ###### keep the different chains #####
    chain_param1[ichain] = thinned21
    chain_param2[ichain] = thinned22
    chain_param3[ichain] = thinned23
    chain_param4[ichain] = thinned24
    chain_param5[ichain] = thinned25
    chain_lik2[ichain] = lik12_mat
    
    chain_param1_all_1[ichain] = c12_mat
    chain_param2_all_1[ichain] = c22_mat
    chain_param3_all_1[ichain] = c32_mat
    chain_param4_all_1[ichain] = c42_mat
    chain_param5_all_1[ichain] = c52_mat
    
var_of_each_chain1 = np.zeros(n_chains)
mean_of_each_chain1 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain1[i] = np.var(chain_param1[i],ddof=1)
    mean_of_each_chain1[i] = np.mean(chain_param1[i])

W1 = np.mean(var_of_each_chain1) 
#mean_of_mean = np.mean(mean_of_each_chain)
B1 = len_chain*(np.var(mean_of_each_chain1,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat12 = (1-1.0/len_chain)*W1+(1.0/len_chain)*B1
R_hat12 = np.sqrt(V_hat12/W1)

    
var_of_each_chain2 = np.zeros(n_chains)
mean_of_each_chain2 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain2[i] = np.var(chain_param2[i],ddof=1)
    mean_of_each_chain2[i] = np.mean(chain_param2[i])

W2 = np.mean(var_of_each_chain2) 
#mean_of_mean = np.mean(mean_of_each_chain)
B2 = len_chain*(np.var(mean_of_each_chain2,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat22 = (1-1.0/len_chain)*W2+(1.0/len_chain)*B2
R_hat22 = np.sqrt(V_hat22/W2)

var_of_each_chain3 = np.zeros(n_chains)
mean_of_each_chain3 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain3[i] = np.var(chain_param3[i],ddof=1)
    mean_of_each_chain3[i] = np.mean(chain_param3[i])

W3 = np.mean(var_of_each_chain3) 
#mean_of_mean = np.mean(mean_of_each_chain)
B3 = len_chain*(np.var(mean_of_each_chain3,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat32 = (1-1.0/len_chain)*W3+(1.0/len_chain)*B3
R_hat32 = np.sqrt(V_hat32/W3)
  
var_of_each_chain4 = np.zeros(n_chains)
mean_of_each_chain4 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain4[i] = np.var(chain_param4[i],ddof=1)
    mean_of_each_chain4[i] = np.mean(chain_param4[i])

W4 = np.mean(var_of_each_chain4) 
#mean_of_mean = np.mean(mean_of_each_chain)
B4 = len_chain*(np.var(mean_of_each_chain4,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat42 = (1-1.0/len_chain)*W4+(1.0/len_chain)*B4
R_hat42 = np.sqrt(V_hat42/W4)

var_of_each_chain5 = np.zeros(n_chains)
mean_of_each_chain5 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain5[i] = np.var(chain_param5[i],ddof=1)
    mean_of_each_chain5[i] = np.mean(chain_param5[i])

W5 = np.mean(var_of_each_chain5) 
#mean_of_mean = np.mean(mean_of_each_chain)
B5 = len_chain*(np.var(mean_of_each_chain5,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat52 = (1-1.0/len_chain)*W5+(1.0/len_chain)*B5
R_hat52 = np.sqrt(V_hat52/W5)

cp_lik1 = thinned21
dp_lik1 = thinned22
noise_lik1 = thinned23
k_lik1 = thinned24
m0_lik1 = thinned25
########################################### KF2 #########################################################
chain_param1 = np.zeros([n_chains,len_chain])
chain_param2 = np.zeros([n_chains,len_chain])
chain_param3 = np.zeros([n_chains,len_chain])
chain_param4 = np.zeros([n_chains,len_chain])
chain_param5 = np.zeros([n_chains,len_chain])
chain_param1_all = np.zeros([n_chains,run])
chain_param2_all = np.zeros([n_chains,run])
chain_param3_all = np.zeros([n_chains,run])
chain_param4_all = np.zeros([n_chains,run])
chain_param5_all = np.zeros([n_chains,run])

chain_lik2 = np.zeros([n_chains,run])
for ichain in range(0,n_chains):
    #c0_minus = np.array([expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0)])    
    #c0_minus = np.array([cp*k_const,dp,noise_const**2,k_const])    
    c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])
    print c0
    c0 = np.log(c0)
    while True:
        try:
            [c2_mat,rate2,lik2_mat,lambda2] = MCMC_BW_adapted_roberts(c0,run,posterior2,lamda2)            
        except ValueError:
            print 'Try new c0'
            c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])
            c0 = np.log(c0)
            print c0
            #[c1_mat,rate1,lik1,lamda_1] = MCMC_BW_adapted_roberts(c0,run,posterior1,lamda1)
            continue
        break
    print "Rate2:  "+str(rate2)
    c2_mat = np.exp(c2_mat)
    #c2_mat.dump("c2_exp_mat_transl_inh_anal_reduced_ode_mylaptop.dat")
    #c2_mat.dump("c2_exp_mat_transl_inh_seed50_expprior_new_roberts3b.dat")
    
    c12_mat = c2_mat[:,0]/c2_mat[:,3]
    c22_mat = c2_mat[:,1]
    c32_mat = c2_mat[:,2]
    c42_mat = c2_mat[:,3]
    c52_mat = c2_mat[:,4]/c2_mat[:,3]

    lik12_mat = lik2_mat[:,0]
    
    plt.figure(157)
    plt.plot(lik12_mat)
    #plt.ylim([-525,-535])
    
    plt.figure(37)
    plt.plot(c12_mat)
    #plt.ylim((75,85))
    plt.figure(38)
    plt.plot(c22_mat)
    plt.figure(39)
    plt.plot(c32_mat)
    #plt.ylim((4.6,5.3))
    plt.figure(40)
    plt.plot(c42_mat)
    #plt.ylim((0.075,0.078))
    plt.figure(41)
    plt.plot(c52_mat)
    
    thinned21 = []
    thinned22 = []
    thinned23 = []
    thinned24 = []
    thinned25 = []
    thinned26 = []
    
    for kept in range(burn_in,run):
        #if (n % 2 == 0):
        thinned21.append(c12_mat[kept])
        thinned22.append(c22_mat[kept])
        thinned23.append(c32_mat[kept])
        thinned24.append(c42_mat[kept])
        thinned25.append(c52_mat[kept])
        
    print "Mean1:  "+str(np.mean(thinned21))
    print "Sigma1: "+str(np.std(thinned21))
    print "Mean2:  "+str(np.mean(thinned22))
    print "Sigma2: "+str(np.std(thinned22))
    print "Mean3:  "+str(np.mean(thinned23))
    print "Sigma3: "+str(np.std(thinned23))
    print "Mean4:  "+str(np.mean(thinned24))
    print "Sigma4: "+str(np.std(thinned24))
    print "Mean5:  "+str(np.mean(thinned25))
    print "Sigma5: "+str(np.std(thinned25))
    
    
    length = range(run)
    ###### keep the different chains #####
    chain_param1[ichain] = thinned21
    chain_param2[ichain] = thinned22
    chain_param3[ichain] = thinned23
    chain_param4[ichain] = thinned24
    chain_param5[ichain] = thinned25
    chain_lik2[ichain] = lik12_mat
    
    chain_param1_all[ichain] = c12_mat
    chain_param2_all[ichain] = c22_mat
    chain_param3_all[ichain] = c32_mat
    chain_param4_all[ichain] = c42_mat
    chain_param5_all[ichain] = c52_mat
    
var_of_each_chain1 = np.zeros(n_chains)
mean_of_each_chain1 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain1[i] = np.var(chain_param1[i],ddof=1)
    mean_of_each_chain1[i] = np.mean(chain_param1[i])

W1 = np.mean(var_of_each_chain1) 
#mean_of_mean = np.mean(mean_of_each_chain)
B1 = len_chain*(np.var(mean_of_each_chain1,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat12 = (1-1.0/len_chain)*W1+(1.0/len_chain)*B1
R_hat12 = np.sqrt(V_hat12/W1)

    
var_of_each_chain2 = np.zeros(n_chains)
mean_of_each_chain2 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain2[i] = np.var(chain_param2[i],ddof=1)
    mean_of_each_chain2[i] = np.mean(chain_param2[i])

W2 = np.mean(var_of_each_chain2) 
#mean_of_mean = np.mean(mean_of_each_chain)
B2 = len_chain*(np.var(mean_of_each_chain2,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat22 = (1-1.0/len_chain)*W2+(1.0/len_chain)*B2
R_hat22 = np.sqrt(V_hat22/W2)

var_of_each_chain3 = np.zeros(n_chains)
mean_of_each_chain3 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain3[i] = np.var(chain_param3[i],ddof=1)
    mean_of_each_chain3[i] = np.mean(chain_param3[i])

W3 = np.mean(var_of_each_chain3) 
#mean_of_mean = np.mean(mean_of_each_chain)
B3 = len_chain*(np.var(mean_of_each_chain3,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat32 = (1-1.0/len_chain)*W3+(1.0/len_chain)*B3
R_hat32 = np.sqrt(V_hat32/W3)
  
var_of_each_chain4 = np.zeros(n_chains)
mean_of_each_chain4 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain4[i] = np.var(chain_param4[i],ddof=1)
    mean_of_each_chain4[i] = np.mean(chain_param4[i])

W4 = np.mean(var_of_each_chain4) 
#mean_of_mean = np.mean(mean_of_each_chain)
B4 = len_chain*(np.var(mean_of_each_chain4,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat42 = (1-1.0/len_chain)*W4+(1.0/len_chain)*B4
R_hat42 = np.sqrt(V_hat42/W4)

var_of_each_chain5 = np.zeros(n_chains)
mean_of_each_chain5 = np.zeros(n_chains)

for i in range(0,n_chains):
    var_of_each_chain5[i] = np.var(chain_param5[i],ddof=1)
    mean_of_each_chain5[i] = np.mean(chain_param5[i])

W5 = np.mean(var_of_each_chain5) 
#mean_of_mean = np.mean(mean_of_each_chain)
B5 = len_chain*(np.var(mean_of_each_chain5,ddof=1))  #variance of the chain means * (samples in each chain)
V_hat52 = (1-1.0/len_chain)*W5+(1.0/len_chain)*B5
R_hat52 = np.sqrt(V_hat52/W5)

###############################################################################################################
plt.figure()
density1 = kde.gaussian_kde(cp_lik1,0.4)
dist_space = np.linspace( min(cp_lik1), max(cp_lik1), 1000 )
plt.hist(cp_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned21,0.4)
dist_space2 = np.linspace( min(thinned21), max(thinned21), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned21,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[0],0,0.02,colors=u'b')
plt.xlabel('$cp$',fontsize=22)
plt.ylabel('$p(cp|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


#plt.figure(17)
#plt.hist(thinned21, 5,normed=True, histtype='step',color = 'g')
#plt.hist(cp_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[0],0,0.05,colors=u'r')
#plt.xlabel(r'$cp$', fontsize=24)
#plt.ylabel(r'posterior', fontsize=24)
#plt.show()

plt.figure()
density1 = kde.gaussian_kde(dp_lik1,0.4)
dist_space = np.linspace( min(dp_lik1), max(dp_lik1), 1000 )
plt.hist(dp_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned22,0.4)
dist_space2 = np.linspace( min(thinned22), max(thinned22), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned22,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[1],0,11,colors=u'b')
plt.xlabel('$dp$',fontsize=22)
plt.ylabel('$p(dp|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

#plt.figure(18)
#plt.hist(thinned22, 5,normed=True, histtype='step',color = 'g')
#plt.hist(dp_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[1],0,80,colors=u'r')
#plt.xlabel(r'$dp$', fontsize=24)
#plt.ylabel(r'posterior', fontsize=24)
#plt.show()

plt.figure()
density1 = kde.gaussian_kde(noise_lik1,0.4)
dist_space = np.linspace( min(noise_lik1), max(noise_lik1), 1000 )
plt.hist(noise_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned23,0.4)
dist_space2 = np.linspace( min(thinned23), max(thinned23), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned23,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[2],0,40,colors=u'b')
plt.xlabel('$noise$',fontsize=22)
plt.ylabel('$p(noise|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

#plt.figure(19)
#plt.hist(thinned23, 5,normed=True, histtype='step',color = 'g')
#plt.hist(noise_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[2],0,0.5,colors=u'r')
#plt.xlabel(r'$noise$', fontsize=24)
#plt.ylabel(r'posterior', fontsize=24)
#plt.show()

plt.figure()
density1 = kde.gaussian_kde(k_lik1,0.4)
dist_space = np.linspace( min(k_lik1), max(k_lik1), 1000 )
plt.hist(k_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned24,0.4)
dist_space2 = np.linspace( min(thinned24), max(thinned24), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned24,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[3],0,200,colors=u'b')
plt.xlabel('$k$',fontsize=22)
plt.ylabel('$p(k|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

#plt.figure(20)
#plt.hist(thinned24, 5,normed=True, histtype='step',color = 'g')
#plt.hist(k_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[3],0,0.05,colors=u'r')
#plt.xlabel(r'$k$', fontsize=24)
#plt.ylabel(r'posterior', fontsize=24)
#plt.show()

plt.figure()
density1 = kde.gaussian_kde(m0_lik1,0.4)
dist_space = np.linspace( min(m0_lik1), max(m0_lik1), 1000 )
plt.hist(m0_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned25,0.4)
dist_space2 = np.linspace( min(thinned25), max(thinned25), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned25,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[4],0,0.01,colors=u'b')
plt.xlabel('$m0$',fontsize=22)
plt.ylabel('$p(m0|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

#plt.figure(21)
#plt.hist(thinned25, 5,normed=True, histtype='step',color = 'g')
#plt.hist(m0_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[4],0,0.05,colors=u'r')
#plt.xlabel(r'$m0$', fontsize=24)
#plt.ylabel(r'posterior', fontsize=24)
#plt.show()

mean12_mat = np.zeros(len(c32_mat))
for i in range(0,len(c32_mat)):
    mean12_mat[i] = np.mean(c32_mat[0:i+1])

plt.figure(22)
plt.plot(mean12_mat)
plt.show()

mean22_mat = np.zeros(len(c22_mat))
for i in range(0,len(c22_mat)):
    mean22_mat[i] = np.mean(c22_mat[0:i+1])
plt.figure(23)
plt.plot(mean22_mat)
plt.show()

mean32_mat = np.zeros(len(c32_mat))
for i in range(0,len(c32_mat)):
    mean32_mat[i] = np.mean(c32_mat[0:i+1])
plt.figure(24)
plt.plot(mean32_mat)
plt.show()

mean42_mat = np.zeros(len(c42_mat))
for i in range(0,len(c42_mat)):
    mean42_mat[i] = np.mean(c42_mat[0:i+1])
plt.figure(25)
plt.plot(mean42_mat)
plt.show()

mean52_mat = np.zeros(len(c52_mat))
for i in range(0,len(c52_mat)):
    mean52_mat[i] = np.mean(c52_mat[0:i+1])
plt.figure(26)
plt.plot(mean52_mat)
plt.show()


#####################################  checking prior vs posterior ################################################################
#r = norm.rvs(200*0.06,1.0, size=1000)
#plt.figure()
#density = kde.gaussian_kde(r)
#dist_space = np.linspace( min(r), max(r), 1000 )
#plt.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density(dist_space),label ="prior")
#density3 = kde.gaussian_kde(thinned25)
#dist_space3 = np.linspace( min(thinned25), max(thinned25), 1000 )
#post = plt.plot( dist_space3, density3(dist_space3) ,label ="posterior3")
#plt.hist(thinned25,normed=True, histtype='stepfilled', alpha=0.2)
#plt.show()

r = expon.rvs(loc=0, scale=10000, size=1000)
plt.figure()
density = kde.gaussian_kde(r)
dist_space = np.linspace( min(r), max(r), 1000 )
plt.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density(dist_space),label ="prior")
#density3 = kde.gaussian_kde(thinned25)
#dist_space3 = np.linspace( min(thinned25), max(thinned25), 1000 )
#post = plt.plot( dist_space3, density3(dist_space3) ,label ="posterior3")
#plt.hist(thinned25,normed=True, histtype='stepfilled', alpha=0.2)
#plt.show()

################## saving data ###############
#c1_mat.dump("c1_mat_exp_transl.dat")
#c2_mat.dump("c2_mat_exp_transl.dat")
#
#
#with open("c1_mat_expTransl.bin", "wb") as output1:
#    pickle.dump(c1_mat, output1)
#    
#with open("c2_mat_expTransl.bin", "wb") as output2:
#    pickle.dump(c2_mat, output2)    