#==============================================================================
#Lotka Volterra (LV) example
#likelihood 1 gives LV likelihood using KF1
#likelihood 2 gives LV likelihood using KF2
#Runs adaptive MCMC 
#==============================================================================
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from deriv_LV import deriv_LV
from LV_jac import LV_jac
from LV2_jac import LV2_jac
from deriv_LV2 import deriv_LV2
#from gil_integr_LV2 import gil_integr_LV2
#from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from MCMC_BW_adapted_roberts import MCMC_BW_adapted_roberts
import random
from scipy.stats import kde
from statsmodels.graphics import tsaplots
from scipy.stats import gamma
import pickle

random.seed(50)
np.random.seed(50)

n = 20 # do not exceed too much, danger of extincted species 
length_of_interval = n
aggr_step =  2.0 # aggregation period
m01 = 10 #initial number of prey species
m02 = 100 #initial number of predator species
c1 = 0.5 #prey production rate
c2 = 0.0025 #predator production rate
c3 = 0.3 #predator death rate

#total iterations and burn in for MCMC
run = 30000
burn_in = 10000

print "aggr_step"+str(aggr_step)
num_rep = 40 # number of independent samples
s_noise = 3.0#9.0 ###  standard deviation of Gaussian noise

###################### import LV dataset ################################
with open("obs_lv_seed50.bin", "rb") as data1:
   obs = pickle.load(data1)

with open("obsavg_lv_seed50.bin", "rb") as data2:
   obs_avg = pickle.load(data2)

#obs = np.load("C:\data_lv_thesis\obs_lvseed50.dat")
#obs_avg = np.load("C:\data_lv_thesis\obsavg_lvseed50.dat")

###################### create dataset ###################################
#real_obs1 = np.zeros((num_rep,(n/aggr_step)))
#real_obs2 = np.zeros((num_rep,(n/aggr_step)))
#obs = np.zeros((num_rep,(n/aggr_step)))
#obs_avg = np.zeros((num_rep,(n/aggr_step)))
#s_noise = 3.0#9.0
#for rep in range(0,num_rep):
#    [obs_trapz,real_obs, obs_integr,time1,species] = gil_integr_LV2(m01,m02,n,aggr_step,c1,c2,c3)
#    obs1 = obs_trapz[:,0]
#    real1 = real_obs[:,0]
#    real2 = real_obs[:,1]
#    noise = np.matrix(np.random.normal(0,s_noise,len(obs1))).T
#    obs_avg1 = obs1/aggr_step+noise
#    obs1 = obs1+noise
#    real_obs1[rep,:] = real1.T
#    real_obs2[rep,:] = real2.T
#    obs[rep,:] = obs1.T
#    obs_avg[rep,:] = obs_avg1.T

P = np.matrix([1,0])
V = np.matrix([s_noise**2]) ### use variance 
m0 = np.matrix([m01,m02])
S0 = np.matrix([[0.1,0],[0,0.1]])
P0 = np.matrix([[1,0],[0,1]])
V0 = np.matrix([[0,0],[0,0]])

c_real = np.array([0.5, 0.0025, 0.3])

mean0 = P0*m0.T
cov0 = P0*S0*P0.T+V0
y0 = np.random.multivariate_normal(np.array(mean0.T)[0],np.array(cov0))
y0 = np.matrix(y0)

lik0 = multivariate_normal.pdf(y0, mean = np.array(mean0.T)[0], cov = np.array(cov0))
llik0 = multivariate_normal.logpdf(y0, mean = np.array(mean0.T)[0], cov = np.array(cov0))

part1_0 = S0*P0.T*(P0*S0*P0.T+V0).I
m0_star = m0.T + part1_0*(y0.T-P0*m0.T)
S0_star = S0 - part1_0*P0*S0


def likelihood1(c):

    c = np.exp(c)
    sum_all = 0
    for r in range(0,num_rep):

        init = np.array([m0_star[(0,0)],m0_star[(1,0)],S0_star[(0,0)],S0_star[(0,1)],S0_star[(1,1)]])
        
        j = 0
        step = aggr_step
        dt = 0.001
        prodl = lik0
        suml = llik0
        for i in np.arange(0,n,step):
            time = np.arange(i,i+step+dt,dt)
            y,info = odeint(deriv_LV,init,time,Dfun=LV_jac,args = (c,),full_output = True)#,rtol=1e-3,atol=1e-3)#,mxstep=500,mxords=1,rtol=1e-4)
            l = len(y)
             
            m = np.matrix([y[l-1,0],y[l-1,1]])
            S = np.matrix([[y[l-1,2],y[l-1,3]],[y[l-1,3],y[l-1,4]]])
            mean1 = P*m.T
            cov1 = P*S*P.T+V
            try:
                multivariate_normal.pdf(obs_avg[r,j], mean=mean1, cov=cov1)
            except ValueError:
                print "c1:  "+str(c)
                print "value1:  "+str(obs_avg[r,j])
                print "r1:  "+str(r)
                print "j1:  "+str(j)
                print "mean1:  "+str(mean1)
                print "cov1:  "+str(cov1)
                print "init1:  "+str(init)
               ##exit
            lik = multivariate_normal.pdf(obs_avg[r,j], mean=mean1, cov=cov1)
            llik = multivariate_normal.logpdf(obs_avg[r,j], mean=mean1, cov=cov1)
            suml += llik
            prodl *= lik
            
        
            part1 = S*P.T*(P*S*P.T+V).I
            m_star = m.T + part1*(obs_avg[r,j].T-P*m.T)
            for spec in range(0,2):
                if m_star[(spec,0)]<0:
                    m_star[(spec,0)] = 0
            S_star = S - part1*P*S
            init = np.array([m_star[(0,0)],m_star[(1,0)],S_star[(0,0)],S_star[(0,1)],S_star[(1,1)]])    
            j +=1
        sum_all = sum_all + suml
        #print 'sum_all'+str(sum_all)
    return sum_all

def likelihood2(c) :
    c = np.exp(c)
    sum_all = 0
    for r in range(0,num_rep):

        init = np.array([m0_star[(0,0)],m0_star[(1,0)],S0_star[(0,0)],S0_star[(0,1)],S0_star[(1,1)],0*m0_star[(0,0)],0*m0_star[(1,0)],0*S0_star[(0,0)],0*S0_star[(0,1)],0*S0_star[(0,1)],0*S0_star[(1,1)],0*S0_star[(0,0)],0*S0_star[(0,1)],0*S0_star[(1,1)]])
    
        j = 0
        step = aggr_step
        dt = 0.001
        prodl = lik0
        suml = llik0
        for i in np.arange(0, n, step):
            time = np.arange(i, i+step+dt, dt)
            y,info = odeint(deriv_LV2, init, time, Dfun=LV2_jac, args = (c,),full_output = True)#,rtol=1e-5,atol=1e-5)
            l = len(y)
          
            mx = np.matrix([y[l-1, 0], y[l-1, 1]])
            S = np.matrix([[y[l-1, 2], y[l-1, 3]], [y[l-1, 3], y[l-1, 4]]])
    
            m = np.matrix([y[l-1, 5], y[l-1, 6]])
            Q = np.matrix([[y[l-1, 11], y[l-1, 12]], [y[l-1, 12], y[l-1, 13]]])
            QM = np.matrix([[y[l-1, 7], y[l-1, 8]], [y[l-1, 9], y[l-1, 10]]])
    
            mean1 = P*m.T
            cov1 = P*Q*P.T+V
            try:
                multivariate_normal.pdf(obs[r,j], mean=mean1, cov=cov1)
            except ValueError:
                print "c:  "+str(c)
                print "value:  "+str(obs[r,j])
                print "r:  "+str(r)
                print "j:  "+str(j)
                print "mean2:  "+str(mean1)
                print "cov2:  "+str(cov1)
                print "init2:  "+str(init)
                
            lik = multivariate_normal.pdf(obs[r,j], mean=mean1, cov=cov1)
            llik = multivariate_normal.logpdf(obs[r,j], mean=mean1, cov=cov1)
            suml += llik
            prodl *= lik

            part1x = QM.T*P.T*(P*Q*P.T+V).I
            mx_star = mx.T + part1x*(obs[r,j].T-P*m.T)
            for spec in range(0,2):
                if mx_star[(spec,0)]<0:
                    mx_star[(spec,0)] = 0
            S_star = S - part1x*P*QM
    
            init = np.array([mx_star[(0,0)],mx_star[(1,0)],S_star[(0,0)],S_star[(0,1)],S_star[(1,1)],0*mx_star[(0,0)],0*mx_star[(1,0)],0*S_star[(0,0)],0*S_star[(0,1)],0*S_star[(0,1)],0*S_star[(1,1)],0*S_star[(0,0)],0*S_star[(0,1)],0*S_star[(1,1)]])
    
            j +=1
        sum_all = sum_all + suml
    return sum_all


def neg_likelihood1(c):
    return -likelihood1(c)

def neg_likelihood2(c):
    return -likelihood2(c)

def posterior1(c):
    return likelihood1(c)+gamma.logpdf(float(np.exp(c[0])),2.0,loc = 0., scale = 1.0/10.0)+gamma.logpdf(float(np.exp(c[1])),2.0,loc = 0., scale = 1.0/10.0)+gamma.logpdf(float(np.exp(c[2])),2.0,loc = 0., scale = 1.0/10.0)+c[0]+c[1]+c[2]

def posterior2(c):
    #return likelihood2(c)+c[0]+c[1]+c[2]
    return likelihood2(c)+gamma.logpdf(float(np.exp(c[0])),2.0,loc = 0., scale = 1.0/10.0)+gamma.logpdf(float(np.exp(c[1])),2.0,loc = 0., scale = 1.0/10.0)+gamma.logpdf(float(np.exp(c[2])),2.0,loc = 0., scale = 1.0/10.0)+c[0]+c[1]+c[2]

########################## Nelder-Mead ###############################
#c0 = np.array([0.5,0.0025,0.3])
#print c0
#c0 = np.log(c0)
#optim1 = minimize(neg_likelihood1,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
#print optim1
#print np.exp(optim1.x)

#print c0
#optim2 = minimize(neg_likelihood2,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
#print optim2
#print np.exp(optim2.x)
##########################################################################
 
#################################################### MCMC1 ############################

c0 = np.array([random.uniform(0.1,1),random.uniform(0,0.01),random.uniform(0.1,1)])
#c0 = np.array([random.uniform(0,1),random.uniform(0,0.1),random.uniform(0,1)])

print c0
c0 = np.log(c0)
print c0
lamda1 = 0.1#1.0#0.1
while True:
    try:
        [c1_mat,rate1,lik1,lamda_1] = MCMC_BW_adapted_roberts(c0,run,posterior1,lamda1)
    except ValueError:
        print 'Try new c0'
        c0 = np.array([random.uniform(0.1,1),random.uniform(0,0.01),random.uniform(0.1,1)])
        c0 = np.log(c0)
        print c0
        #[c1_mat,rate1,lik1,lamda_1] = MCMC_BW_adapted_roberts(c0,run,posterior1,lamda1)
        continue
    break

#c1_mat.dump("c1_mat_log_lv_seed50_mylaptop.dat")

c12_mat = np.exp(c1_mat[:,0])
tsaplots.plot_acf(c12_mat)

c22_mat = np.exp(c1_mat[:,1])
c32_mat = np.exp(c1_mat[:,2])

plt.figure(7)
plt.plot(c12_mat)
plt.ylim((0.45,0.51))
plt.figure(8)
plt.plot(c22_mat)
plt.figure(9)
plt.plot(c32_mat)

thinned21 = []
thinned22 = []
thinned23 = []
for kept in range(burn_in,run):
    #if (n % 2 == 0):
    thinned21.append(c12_mat[kept])
    thinned22.append(c22_mat[kept])
    thinned23.append(c32_mat[kept])
        
print "Mean1:  "+str(np.mean(thinned21))
print "Sigma1: "+str(np.std(thinned21))
print "Mean2:  "+str(np.mean(thinned22))
print "Sigma2: "+str(np.std(thinned22))
print "Mean3:  "+str(np.mean(thinned23))
print "Sigma3: "+str(np.std(thinned23))


mean12_mat = np.zeros(len(c32_mat))
for i in range(0,len(c32_mat)):
    mean12_mat[i] = np.mean(c32_mat[0:i+1])

plt.figure(12)
plt.plot(mean12_mat)

mean22_mat = np.zeros(len(c22_mat))
for i in range(0,len(c22_mat)):
    mean22_mat[i] = np.mean(c22_mat[0:i+1])
plt.figure(13)
plt.plot(mean22_mat)

mean32_mat = np.zeros(len(c32_mat))
for i in range(0,len(c32_mat)):
    mean32_mat[i] = np.mean(c32_mat[0:i+1])
plt.figure(14)
plt.plot(mean32_mat)
plt.show()
gr_lik1 = thinned21
kp_lik1 = thinned22
gp_lik1 = thinned23

############################################ MCMC 2 ################################################
lamda2 = 0.1#1.0#0.1
#c0 = np.array([random.uniform(0,1),random.uniform(0,0.01),random.uniform(0,1)])
print c0
#c0 = np.log(c0)
while True:
    try:
        [c2_mat,rate2,lik2,lamda_2] = MCMC_BW_adapted_roberts(c0,run,posterior2,lamda2)
    except ValueError:
        print 'Try new c0'
        c0 = np.array([random.uniform(0.1,1),random.uniform(0,0.01),random.uniform(0.1,1)])
        c0 = np.log(c0)
        print c0
        continue
    break


c12_mat = np.exp(c2_mat[:,0])
c22_mat = np.exp(c2_mat[:,1])
c32_mat = np.exp(c2_mat[:,2])

step = aggr_step
plt.figure(7)
plt.plot(c12_mat)
plt.figure(8)
plt.plot(c22_mat)
plt.figure(9)
plt.plot(c32_mat)

thinned21 = []
thinned22 = []
thinned23 = []
for keep in range(burn_in,run):
    #if (n % 2 == 0):
    thinned21.append(c12_mat[keep])
    thinned22.append(c22_mat[keep])
    thinned23.append(c32_mat[keep])
        
print "Mean1:  "+str(np.mean(thinned21))
print "Sigma1: "+str(np.std(thinned21))
print "Mean2:  "+str(np.mean(thinned22))
print "Sigma2: "+str(np.std(thinned22))
print "Mean3:  "+str(np.mean(thinned23))
print "Sigma3: "+str(np.std(thinned23))

#
#plt.figure(10)
#plt.hist(thinned21, 5, histtype='step',color = 'g')
#plt.hist(gr_lik1, 5, histtype='step',color = 'b')
#plt.vlines(c_real[0],0,700,colors=u'r')
#plt.xlabel(r'$c1$', fontsize=24)
#plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
#plt.show()
#
#plt.figure(11)
#plt.hist(thinned22, 5, histtype='step',color = 'g')
#plt.hist(kp_lik1, 5, histtype='step',color = 'b')
#plt.vlines(c_real[1],0,500,colors=u'r')
#plt.xlabel(r'$c2$', fontsize=24)
#plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
#plt.show()
#
#plt.figure(15)
#plt.hist(thinned23, 5, histtype='step',color = 'g')
#plt.hist(gp_lik1, 5, histtype='step',color = 'b')
#plt.vlines(c_real[2],0,500,colors=u'r')
#plt.xlabel(r'$c3$', fontsize=24)
#plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
#plt.show()
mean12_mat = np.zeros(len(c32_mat))
for i in range(0,len(c32_mat)):
    mean12_mat[i] = np.mean(c32_mat[0:i+1])

plt.figure(12)
plt.plot(mean12_mat)

mean22_mat = np.zeros(len(c22_mat))
for i in range(0,len(c22_mat)):
    mean22_mat[i] = np.mean(c22_mat[0:i+1])
plt.figure(13)
plt.plot(mean22_mat)

mean32_mat = np.zeros(len(c32_mat))
for i in range(0,len(c32_mat)):
    mean32_mat[i] = np.mean(c32_mat[0:i+1])
plt.figure(14)
plt.plot(mean32_mat)


plt.figure()
density1 = kde.gaussian_kde(gr_lik1,0.4)
dist_space = np.linspace( min(gr_lik1), max(gr_lik1), 1000 )
plt.hist(gr_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned21,0.4)
dist_space2 = np.linspace( min(thinned21), max(thinned21), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned21,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[0],0,70,colors=u'b')
plt.xlabel('$c_1$',fontsize=22)
plt.ylabel('$p(c_1|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
density1 = kde.gaussian_kde(kp_lik1)
dist_space = np.linspace( min(kp_lik1), max(kp_lik1), 1000 )
plt.hist(kp_lik1, normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned22)
dist_space2 = np.linspace( min(thinned22), max(thinned22), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned22,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[1],0,10000,colors=u'b')
plt.xlabel('$c_2$',fontsize=22)
plt.ylabel('$p(c_2|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
density1 = kde.gaussian_kde(gp_lik1)
dist_space = np.linspace( min(gp_lik1), max(gp_lik1), 1000 )
plt.hist(gp_lik1, normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color = 'r')
density2 = kde.gaussian_kde(thinned23)
dist_space2 = np.linspace( min(thinned23), max(thinned23), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned23,normed=True, histtype='stepfilled', alpha=0.2)
plt.vlines(c_real[2],0,40,colors=u'b')
plt.xlabel('$c_3$',fontsize=22)
plt.ylabel('$p(c_3|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

#c1_mat.dump("c1_mat_log_lv.dat")
#c2_mat.dump("c2_mat_log_lv.dat")


#with open("c1_mat_logLV.bin", "wb") as output1:
#    pickle.dump(c1_mat, output1)
#    
#with open("c2_mat_logLV.bin", "wb") as output2:
#    pickle.dump(c2_mat, output2)    