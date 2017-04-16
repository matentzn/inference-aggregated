#==============================================================================
#Reproduces the plots of the Lotka Volterra (LV) example
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from statsmodels.graphics import tsaplots
#from scipy.stats import pearsonr
import pickle 

run = 30000
burn_in = 10000

c_real = np.array([0.5, 0.0025, 0.3])

#c1_all_log = np.load("/home/owl/Desktop/paper_code/c1_mat_log_lv.dat")
#c2_all_log = np.load("/home/owl/Desktop/paper_code/c2_mat_log_lv.dat")

with open("c1_mat_logLV.bin", "rb") as data1:
   c1_all_log = pickle.load(data1)
with open("c2_mat_logLV.bin", "rb") as data2:
   c2_all_log = pickle.load(data2)

c11_mat = np.exp(c1_all_log[:,0])
c21_mat = np.exp(c1_all_log[:,1])
c31_mat = np.exp(c1_all_log[:,2])

tsaplots.plot_acf(c11_mat)
tsaplots.plot_acf(c21_mat)
tsaplots.plot_acf(c31_mat)

plt.figure(7)
plt.plot(np.log(c11_mat))
plt.ylabel(r'$log(\theta_1)$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.title('KF1-adaptive MCMC')
#plt.title('KF1-rwMH')
plt.ylim((-0.8,-0.6))

plt.figure(8)
plt.plot(np.log(c21_mat))
plt.ylabel(r'$log(\theta_2)$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.title('KF1-adaptive MCMC')
#plt.title('KF1-rwMH')
plt.ylim((-6.2,-5.9))

plt.figure(9)
plt.plot(np.log(c31_mat))
plt.ylabel(r'$log(\theta_3)$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.title('KF1-adaptive MCMC')
#plt.title('KF1-rwMH')
plt.ylim((-1.7,-1.0))

thinned21 = []
thinned22 = []
thinned23 = []
for keep in range(burn_in,run):
    #if (n % 2 == 0):
    thinned21.append(c11_mat[keep])
    thinned22.append(c21_mat[keep])
    thinned23.append(c31_mat[keep])

print "Mean1:  "+str(np.mean(thinned21))
print "Sigma1: "+str(np.std(thinned21))
print "Mean2:  "+str(np.mean(thinned22))
print "Sigma2: "+str(np.std(thinned22))
print "Mean3:  "+str(np.mean(thinned23))
print "Sigma3: "+str(np.std(thinned23))

gr_lik1 = thinned21
kp_lik1 = thinned22
gp_lik1 = thinned23

c12_mat = np.exp(c2_all_log[:,0])
c22_mat = np.exp(c2_all_log[:,1])
c32_mat = np.exp(c2_all_log[:,2])

plt.figure(107)
plt.plot(np.log(c12_mat))
plt.ylabel(r'$log(\theta_1)$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.title('KF2-adaptive MCMC')
#plt.title('KF2-rwMH')
plt.ylim((-0.8,-0.6))
#plt.ylim((0.4,0.7))

plt.figure(108)
plt.plot(np.log(c22_mat))
plt.ylabel(r'$log(\theta_2)$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.title('KF2-adaptive MCMC')
#plt.title('KF2-rwMH')
plt.ylim((-6.1,-5.85))
#plt.ylim((0.0005,0.0035))

plt.figure(109)
plt.plot(np.log(c32_mat))
plt.ylabel(r'$log(\theta_3)$',fontsize=22)
plt.xlabel('runs',fontsize=14)
plt.title('KF2-adaptive MCMC')
#plt.title('KF2-rwMH')
plt.ylim((-1.4,-1.0))


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
plt.xlabel(r'$\theta_1$',fontsize=22)
plt.ylabel(r'$p(\theta_1|data)$',fontsize=18)
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
plt.xlabel(r'$\theta_2$',fontsize=22)
plt.ylabel(r'$p(\theta_2|data)$',fontsize=18)
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
plt.vlines(c_real[2],0,45,colors=u'b')
plt.xlabel(r'$\theta_3$',fontsize=22)
plt.ylabel(r'$p(\theta_3|data)$',fontsize=18)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

#==============================================================================
# Correlation between the parameters
#==============================================================================
#plt.figure()
#plt.plot(thinned21,thinned22)
#plt.xlabel(r'$\theta_1$',fontsize=22)
#plt.ylabel(r'$\theta_2$',fontsize=22)
#print pearsonr(thinned21,thinned22)
#
#plt.figure()
#plt.plot(thinned21,thinned23)
#plt.xlabel(r'$\theta_1$',fontsize=22)
#plt.ylabel(r'$\theta_3$',fontsize=22)
#print pearsonr(thinned21,thinned23)
#
#plt.figure()
#plt.plot(thinned22,thinned23)
#plt.xlabel(r'$\theta_2$',fontsize=22)
#plt.ylabel(r'$\theta_3$',fontsize=22)
#print pearsonr(thinned22,thinned23)
