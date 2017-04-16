#==============================================================================
#Reproduces the plots of the Translation inhibition example with luciferase data
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from scipy.stats import norm
#import triangle 
import random 
#import pandas as pd
random.seed(100)
np.random.seed(100)
import pickle

run = 100000
burn_in = 60000

##########################################################################
with open("c1_3chains_luci_1init.bin", "rb") as data:
   c_all1 = pickle.load(data)

chain1 = np.exp(c_all1[0])
dp_lik1 = chain1[burn_in:run,1]
#==============================================================================
with open("c2_3chains_luci_1init.bin", "rb") as data:
   c_all2 = pickle.load(data)

chain2 = np.exp(c_all2[0])
dp_lik2 = chain2[burn_in:run,1]
##############################################################################
# use c_all1 for KF1 results and c_all2 for KF2 results
c_all = c_all2#c_all1#c_all1+c_all2

n = 8.5#4.5#8.5#9
length_of_interval = n
aggr_step = 0.5

num_rep = 12#37

n_chains = 3#6
len_chain = run-burn_in
lamda2 = 0.1
chain_param1 = np.zeros([n_chains,len_chain])
chain_param2 = np.zeros([n_chains,len_chain])
chain_param3 = np.zeros([n_chains,len_chain])
chain_param4 = np.zeros([n_chains,len_chain])
chain_param5 = np.zeros([n_chains,len_chain])
chain_lik2 = np.zeros([n_chains,run])
for ichain in range(0,n_chains):
    
    c2_mat = c_all[ichain]
    c2_mat = np.exp(c2_mat)
    
    c12_mat = c2_mat[:,0]/c2_mat[:,3]
    c22_mat = c2_mat[:,1]
    c32_mat = c2_mat[:,2]
    c42_mat = c2_mat[:,3]
    c52_mat = c2_mat[:,4]/c2_mat[:,3]
    
    
    plt.figure(37)
    plt.plot(c12_mat)
    plt.ylabel(r'$c_p$',fontsize=22)
    plt.xlabel('runs',fontsize=14)
    plt.figure(38)
    plt.plot(c22_mat)
    plt.ylim((0,2))
    plt.ylabel(r'$d_p$',fontsize=22)
    plt.xlabel('runs',fontsize=14)    
    plt.figure(39)
    plt.plot(c32_mat)
    plt.ylim((-0.001,0.06))   
    plt.ylabel(r'$s$',fontsize=22)
    plt.xlabel('runs',fontsize=14)   
    plt.figure(40)
    plt.plot(c42_mat)
    plt.ylim((0,0.1)) 
    plt.ylabel(r'$k$',fontsize=22)
    plt.xlabel('runs',fontsize=14)       
    plt.figure(41)
    plt.plot(c52_mat)
    plt.ylabel(r'$m_0$',fontsize=22)
    plt.xlabel('runs',fontsize=14)
    
    thinned21 = []
    thinned22 = []
    thinned23 = []
    thinned24 = []
    thinned25 = []
    
    for kept in range(burn_in,run):
        #if (n % 2 == 0):
        thinned21.append(c12_mat[kept])
        thinned22.append(c22_mat[kept])
        thinned23.append(c32_mat[kept])
        thinned24.append(c42_mat[kept])
        thinned25.append(c52_mat[kept])
    print 'new chain'        
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
    
#    for i in range(5,num_rep):
#        print "Mean:  "+str(np.mean(c2_mat[:,i]/c2_mat[:,3]))
#        print "Sigma:  "+str(np.std(c2_mat[:,i]/c2_mat[:,3]))
    
    
    length = range(run)
    ###### keep the different chains #####
    chain_param1[ichain] = thinned21
    chain_param2[ichain] = thinned22
    chain_param3[ichain] = thinned23
    chain_param4[ichain] = thinned24
    chain_param5[ichain] = thinned25
   
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
#density1 = kde.gaussian_kde(cp_lik1,0.4)
#dist_space = np.linspace( min(cp_lik1), max(cp_lik1), 1000 )
#plt.hist(cp_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned21,0.4)
dist_space2 = np.linspace( min(thinned21), max(thinned21), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned21,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.xlabel('$cp$',fontsize=22)
plt.ylabel('$p(cp|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


plt.figure()
#density1 = kde.gaussian_kde(dp_lik1,0.4)
#dist_space = np.linspace( min(dp_lik1), max(dp_lik1), 1000 )
#plt.hist(dp_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned22,0.4)
dist_space2 = np.linspace( min(thinned22), max(thinned22), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned22,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.xlabel('$dp$',fontsize=22)
plt.ylabel('$p(dp|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


plt.figure()
#density1 = kde.gaussian_kde(noise_lik1,0.4)
#dist_space = np.linspace( min(noise_lik1), max(noise_lik1), 1000 )
#plt.hist(noise_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned23,0.4)
dist_space2 = np.linspace( min(thinned23), max(thinned23), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned23,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.xlabel('$s$',fontsize=22)
plt.ylabel('$p(s|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


plt.figure()
#density1 = kde.gaussian_kde(k_lik1,0.4)
#dist_space = np.linspace( min(k_lik1), max(k_lik1), 1000 )
#plt.hist(k_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned24,0.4)
dist_space2 = np.linspace( min(thinned24), max(thinned24), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned24,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.xlabel('$k$',fontsize=22)
plt.ylabel('$p(k|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.figure()
#density1 = kde.gaussian_kde(m0_lik1,0.4)
#dist_space = np.linspace( min(m0_lik1), max(m0_lik1), 1000 )
#plt.hist(m0_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(thinned25,0.4)
dist_space2 = np.linspace( min(thinned25), max(thinned25), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(thinned25,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.xlabel('$m0$',fontsize=22)
plt.ylabel('$p(m0|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()
#mean12_mat = np.zeros(len(c32_mat))
#for i in range(0,len(c32_mat)):
#    mean12_mat[i] = np.mean(c32_mat[0:i+1])
#
#plt.figure(22)
#plt.plot(mean12_mat)
#plt.show()
#
#mean22_mat = np.zeros(len(c22_mat))
#for i in range(0,len(c22_mat)):
#    mean22_mat[i] = np.mean(c22_mat[0:i+1])
#plt.figure(23)
#plt.plot(mean22_mat)
#plt.show()
#
#mean32_mat = np.zeros(len(c32_mat))
#for i in range(0,len(c32_mat)):
#    mean32_mat[i] = np.mean(c32_mat[0:i+1])
#plt.figure(24)
#plt.plot(mean32_mat)
#plt.show()
#
#mean42_mat = np.zeros(len(c42_mat))
#for i in range(0,len(c42_mat)):
#    mean42_mat[i] = np.mean(c42_mat[0:i+1])
#plt.figure(25)
#plt.plot(mean42_mat)
#plt.show()
#
#mean52_mat = np.zeros(len(c52_mat))
#for i in range(0,len(c52_mat)):
#    mean52_mat[i] = np.mean(c52_mat[0:i+1])
#plt.figure(26)
#plt.plot(mean52_mat)
#plt.show()

#r = norm.rvs(2.0,1.0, size=1000)
#c1_transformed = np.zeros([len(thinned21)])
#for i in range(0,len(thinned21)): #for i in range(0,len(c32_mat)): 
#    c1_transformed[i] = np.log(thinned21[i]*thinned24[i])
#  
#plt.figure()
#density = kde.gaussian_kde(r)
#dist_space = np.linspace( min(r), max(r), 1000 )
#plt.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
#prior = plt.plot( dist_space, density(dist_space),label ="prior")
#density3 = kde.gaussian_kde(c1_transformed[:])
#dist_space3 = np.linspace( min(c1_transformed[:]), max(c1_transformed[:]), 1000 )
#post = plt.plot( dist_space3, density3(dist_space3) ,label ="posterior3")
#plt.hist(c1_transformed[:],normed=True, histtype='stepfilled', alpha=0.2)
#plt.show()

r = norm.rvs(-4.0,2.0, size=1000)
c3_transformed = np.log(thinned23)

plt.figure()
density = kde.gaussian_kde(r)
dist_space = np.linspace( min(r), max(r), 1000 )
plt.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density(dist_space),label ="prior")
density3 = kde.gaussian_kde(c3_transformed[:])
dist_space3 = np.linspace( min(c3_transformed[:]), max(c3_transformed[:]), 1000 )
post = plt.plot( dist_space3, density3(dist_space3) ,label ="posterior3")
plt.hist(c3_transformed[:],normed=True, histtype='stepfilled', alpha=0.2)
plt.show()

print "R_hat1:  "+str(R_hat12)
print "R_hat2:  "+str(R_hat22)
print "R_hat3:  "+str(R_hat32)
print "R_hat4:  "+str(R_hat42)
print "R_hat5:  "+str(R_hat52)


plt.figure()
density1 = kde.gaussian_kde(dp_lik1,0.4)
dist_space = np.linspace( min(dp_lik1), max(dp_lik1), 1000 )
plt.hist(dp_lik1, bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
density2 = kde.gaussian_kde(dp_lik2,0.4)
dist_space2 = np.linspace( min(dp_lik2), max(dp_lik2), 1000 )
post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
plt.hist(dp_lik2,bins = 10,normed=True, histtype='stepfilled', alpha=0.2)
plt.xlabel('$dp$',fontsize=22)
plt.ylabel('$p(dp|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlim((0.6,1.7))  
plt.ylim((0,5))         
plt.show()
