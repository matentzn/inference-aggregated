#==============================================================================
#Reproduces the plots of the Translation inhibition example with synthetic data
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
#from statsmodels.graphics import tsaplots
#from scipy.stats import pearsonr
import pickle

run = 30000
burn_in = 10000

m0_1 = 400

cp = 200
dp = 0.97
k_const = 0.03
noise_const = 0.1

c_real = np.array([cp,dp,noise_const,k_const,m0_1]) 

#c1_mat = np.load("/home/owl/Desktop/paper_code/c1_mat_exp_transl.dat")
#c2_mat = np.load("/home/owl/Desktop/paper_code/c2_mat_exp_transl.dat")

with open("c1_mat_expTransl.bin", "rb") as data1:
   c1_mat = pickle.load(data1)
with open("c2_mat_expTransl.bin", "rb") as data2:
   c2_mat = pickle.load(data2)

c11_mat = c1_mat[:,0]/c1_mat[:,3]
c21_mat = c1_mat[:,1]
c31_mat = c1_mat[:,2]
c41_mat = c1_mat[:,3]
c51_mat = c1_mat[:,4]/c1_mat[:,3]



thinned21 = []
thinned22 = []
thinned23 = []
thinned24 = []
thinned25 = []
thinned26 = []

for kept in range(burn_in,run):
    #if (n % 2 == 0):
    thinned21.append(c11_mat[kept])
    thinned22.append(c21_mat[kept])
    thinned23.append(c31_mat[kept])
    thinned24.append(c41_mat[kept])
    thinned25.append(c51_mat[kept])
    
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

cp_lik1 = thinned21
dp_lik1 = thinned22
noise_lik1 = thinned23
k_lik1 = thinned24
m0_lik1 = thinned25

c12_mat = c2_mat[:,0]/c2_mat[:,3]
c22_mat = c2_mat[:,1]
c32_mat = c2_mat[:,2]
c42_mat = c2_mat[:,3]
c52_mat = c2_mat[:,4]/c2_mat[:,3]


plt.figure(37)
plt.plot(c11_mat,label ="KF1",color = 'b')
plt.plot(c12_mat,label ="KF2",color = 'g')
#plt.ylim((75,85))
plt.ylabel(r'$c_p$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.figure(38)
plt.plot(c21_mat,label ="KF1",color = 'b')
plt.plot(c22_mat,label ="KF2",color = 'g')
plt.ylabel(r'$d_p$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.figure(39)
plt.plot(c31_mat,label ="KF1",color = 'b')
plt.plot(c32_mat,label ="KF2",color = 'g')
plt.ylabel(r'$s$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylim((0,0.5))

plt.figure(40)
plt.plot(c41_mat,label ="KF1",color = 'b')
plt.plot(c42_mat,label ="KF2",color = 'g')
plt.xlabel('runs',fontsize=14)
plt.ylabel(r'$k$',fontsize=22)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylim((0,0.1))

plt.figure(41)
plt.plot(c51_mat,label ="KF1",color = 'b')
plt.plot(c52_mat,label ="KF2",color = 'g')
plt.xlabel('runs',fontsize=14)
plt.ylabel(r'$m_0$',fontsize=22)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

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
plt.xlabel('$s$',fontsize=22)
plt.ylabel('$p(noise|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()


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
