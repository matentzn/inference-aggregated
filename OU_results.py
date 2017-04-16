import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import pickle 

run = 50000
burn_in = 30000#45000

#c1_1_all_log = np.load("/home/owl/Desktop/paper_code/c1_mat1_all01.dat")
#c2_1_all_log = np.load("/home/owl/Desktop/paper_code/c2_mat1_all01.dat")
#c1_2_all_log = np.load("/home/owl/Desktop/paper_code/c1_mat2_all01.dat")
#c2_2_all_log = np.load("/home/owl/Desktop/paper_code/c2_mat2_all01.dat")

with open("c1_mat1_all_step2.bin", "rb") as data1:
   c1_1_all_log = pickle.load(data1)
with open("c2_mat1_all_step2.bin", "rb") as data2:
   c2_1_all_log = pickle.load(data2)
with open("c1_mat2_all_step2.bin", "rb") as data3:
   c1_2_all_log = pickle.load(data3)
with open("c2_mat2_all_step2.bin", "rb") as data4:
   c2_2_all_log = pickle.load(data4)

c1_1_all_exp = np.exp(c1_1_all_log)
c2_1_all_exp = np.exp(c2_1_all_log)
c1_2_all_exp = np.exp(c1_2_all_log)
c2_2_all_exp = np.exp(c2_2_all_log)


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



print('KF1:')
print np.mean(c1_1_all_exp[:,burn_in:run])
print np.std(c1_1_all_exp[:,burn_in:run])
print np.mean(c2_1_all_exp[:,burn_in:run])
print np.std(c2_1_all_exp[:,burn_in:run])

print('KF2:')
print np.mean(c1_2_all_exp[:,burn_in:run])
print np.std(c1_2_all_exp[:,burn_in:run])
print np.mean(c2_2_all_exp[:,burn_in:run])
print np.std(c2_2_all_exp[:,burn_in:run])

rep = len(c1_1_all_log)

accept = 0
for k in range(0,run-1):
    if c1_1_all_log[0,k] != c1_1_all_log[0,k+1] or c2_1_all_log[0,k] != c2_1_all_log[0,k+1]:
        accept += 1
        
len_chain = run-burn_in
stat_var1_all = np.zeros((rep,len_chain))
stat_var2_all = np.zeros((rep,len_chain))
c_real = [4.0,2.0]
for i in range(0,rep):
    
    c1_log = c1_1_all_log[i]
    c2_log = c2_1_all_log[i]   
    #c1_mat2_all_log.dump("c1_mat2_all_log_step1.dat")
    #c2_mat2_all_log.dump("c2_mat2_all_log_step1.dat")
    c1_log2 = c1_2_all_log[i]
    c2_log2 = c2_2_all_log[i]   
    
    c1_mat1 = np.exp(c1_log)
    c2_mat1 = np.exp(c2_log)     
    c1_mat2 = np.exp(c1_log2)
    c2_mat2 = np.exp(c2_log2)
            
    thinned1_1 = []
    thinned2_1 = []
    stationary_var1 = []    
    
    thinned1 = []
    thinned2 = []
    stationary_var2 = []
    for kept_samples in range(burn_in,run):
        #if (n % 2 == 0):
        thinned1_1.append(c1_mat1[kept_samples])
        thinned2_1.append(c2_mat1[kept_samples])
        stationary_var1.append(0.5*((c2_mat1[kept_samples])**2)/c1_mat1[kept_samples])

        thinned1.append(c1_mat2[kept_samples])
        thinned2.append(c2_mat2[kept_samples])
        stationary_var2.append(0.5*((c2_mat2[kept_samples])**2)/c1_mat2[kept_samples])
    
    print "KF1, dataset:  "+str(i+1)  
    print "Mean1:  "+str(np.mean(thinned1_1))
    print "Sigma1: "+str(np.std(thinned1_1))
    print "Mean2:  "+str(np.mean(thinned2_1))
    print "Sigma2: "+str(np.std(thinned2_1))
    print "Mean3:  "+str(np.mean(stationary_var1))
    print "Sigma3: "+str(np.std(stationary_var1))
    
    print "KF2, dataset:  "+str(i+1)             
    print "Mean1:  "+str(np.mean(thinned1))
    print "Sigma1: "+str(np.std(thinned1))
    print "Mean2:  "+str(np.mean(thinned2))
    print "Sigma2: "+str(np.std(thinned2))
    print "Mean3:  "+str(np.mean(stationary_var2))
    print "Sigma3: "+str(np.std(stationary_var2))
    #c1_mat2_all[epanalipsi,:] = thinned1
    #c2_mat2_all[epanalipsi,:] = thinned2
    stat_var1_all[i,:] = stationary_var1
    stat_var2_all[i,:] = stationary_var2

    length = range(run)
    plt.figure(23)
    plt.plot(c1_mat1)
    plt.xlabel('runs', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=24)
    plt.title('KF1', fontsize=24)
    plt.ylim((0,5))
    plt.show()

    plt.figure(24)
    plt.plot(c2_mat1)
    plt.xlabel('runs', fontsize=16)
    plt.ylabel(r'$\sigma$', fontsize=24)
    plt.title('KF1', fontsize=24)
    plt.ylim((0,3))
    plt.show()

    
    plt.figure(25)
    plt.plot(c1_mat2)
    plt.xlabel('runs', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=24)
    plt.title('KF2', fontsize=24)
    plt.ylim((1,6))   
    plt.show()

     
    plt.figure(26)
    plt.plot(c2_mat2)
    plt.xlabel('runs', fontsize=16)
    plt.ylabel(r'$\sigma$', fontsize=24)
    plt.title('KF2', fontsize=24)
    plt.ylim((1,4))
    plt.show()

    
    
#    mean_c12 = np.zeros(len(c1_mat2))
#    
#    for k in range(0,len(c1_mat2)) :
#        mean_c12[k] = np.mean(c1_mat2[0:k+1])
#    plt.figure(24)
#    plt.plot(length,mean_c12)
#    plt.figure(25)
#    plt.plot(c2_mat2)
#    
#    mean_c22 = np.zeros(len(c2_mat2))
#    
#    for k in range(0,len(c2_mat2)) :
#        mean_c22[k] = np.mean(c2_mat2[0:k+1])
#    plt.figure(26)
#    plt.plot(length,mean_c22)
    
    plt.figure(22)
    plt.hist(stationary_var1, 5, histtype='step')    
    plt.hist(stationary_var2, 5, histtype='step')
    plt.vlines(0.5,0,1000.0,colors=u'r')
    plt.xlabel(r'$v_1$', fontsize=24)
    
    drift1 = thinned1_1
    diffus1 = thinned2_1


    plt.figure()
    density1 = kde.gaussian_kde(drift1)
    dist_space = np.linspace( min(drift1), max(drift1), 1000 )
    plt.hist(drift1, normed=True, histtype='stepfilled', alpha=0.2)
    prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
    density2 = kde.gaussian_kde(thinned1)
    dist_space2 = np.linspace( min(thinned1), max(thinned1), 1000 )
    post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
    plt.hist(thinned1,normed=True, histtype='stepfilled', alpha=0.2)
    plt.vlines(c_real[0],0,2.55,colors=u'b')
    plt.xlabel(r'$\alpha$',fontsize=22)
    plt.ylabel(r'$p( \alpha |data)$',fontsize=18)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
    plt.figure()
    density1 = kde.gaussian_kde(diffus1)
    dist_space = np.linspace( min(diffus1), max(diffus1), 1000 )
    plt.hist(diffus1, normed=True, histtype='stepfilled', alpha=0.2)
    prior = plt.plot( dist_space, density1(dist_space),label ="KF1",color ='r')
    density2 = kde.gaussian_kde(thinned2)
    dist_space2 = np.linspace( min(thinned2), max(thinned2), 1000 )
    post = plt.plot( dist_space2, density2(dist_space2) ,label ="KF2",color ='g')
    plt.hist(thinned2,normed=True, histtype='stepfilled', alpha=0.2)
    plt.vlines(c_real[1],0,2.5,colors=u'b')
    plt.xlabel(r'$\sigma$',fontsize=22)
    plt.ylabel(r'$p( \sigma |data)$',fontsize=18)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

 
#    plt.figure(21)
#    plt.hist(thinned1, 5,normed=True, histtype='step',color = 'g')
#    plt.hist(drift1, 5,normed=True, histtype='step',color = 'b')
#    plt.vlines(c_real[0],0,2.0,colors=u'r')
#    plt.xlabel(r'$\alpha$', fontsize=24)
#    #plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
#    plt.show()
#    
#    plt.figure(22)
#    plt.hist(thinned2, 5,normed=True, histtype='step',color = 'g')
#    plt.hist(diffus1, 5,normed=True, histtype='step',color = 'b')
#    plt.vlines(c_real[1],0,2.0,colors=u'r')
#    plt.xlabel(r'$\sigma$', fontsize=24)
#    #plt.ylabel(r'$\cal L($Data$;\theta)$', fontsize=24)
#    plt.show()

#stat_var1_all.dump("stat_var1_all4.dat")
#stat_var2_all.dump("stat_var2_all4.dat")


#print('Stationary variance1:')
#print np.mean(stat_var1_all)
#print np.std(stat_var1_all)
#
#print('Stationary variance2:')
#print np.mean(stat_var2_all)
#print np.std(stat_var2_all)
#
#
#print('Stationary variance1 percentiles:')
#print np.percentile(stat_var1_all,50)
#print np.percentile(stat_var1_all,75)
#print np.percentile(stat_var1_all,25)
#
#print('Stationary variance2 percentiles:')
#print np.percentile(stat_var2_all,50)
#print np.percentile(stat_var2_all,75)
#print np.percentile(stat_var2_all,25)
#
#
#
##plt.figure(10)
##plt.boxplot([stat_var01[i],stat_var05[i],stat_var1[i],stat_var2[i],stat_var4[i]], showfliers = False)#,whis="range")
##plt.axhline(y=0.5,color='k',ls='dashed')    
##plt.ylabel('stat_var')
##plt.xlabel('$\Delta$')
#
##kf1 = np.array([0.590196078391,0.414762367364,0.265443780122,0.129459927815])
##kf1_var = np.array([0.0959332356848,0.0631135680258,0.0373826089167,0.0189414142309])
###
##kf2 = np.array([0.56289174933,0.651075673836,0.650444671812,0.572354375569])
##kf2_var = np.array([0.086722127014,0.0886029935712,0.104639199562,0.113303076863])    
#
#kf1 = np.array([0.590196078391,0.414762367364,0.265443780122,0.129459927815])
#kf1_var = 2*np.array([0.0959332356848,0.0631135680258,0.0373826089167,0.0189414142309])
##
#kf2 = np.array([0.56289174933,0.651075673836,0.650444671812,0.572354375569])
#kf2_var = 2*np.array([0.086722127014,0.0886029935712,0.104639199562,0.113303076863])    
#
#
#x = np.array([0.1,0.5,1.0,2.0])
##x = np.array([0.1,0.5,1.0,2.0,4.0])
##
#plt.figure(10)
#plt.errorbar(x,kf1,yerr = kf1_var,fmt='o',label = 'KF1')
#plt.axhline(y=0.5,color='k',ls='dashed')    
#plt.ylim((0,1.0))
#plt.xlim((0,2.2))
#plt.ylabel('stat_var')
#plt.xlabel('$\Delta$')
#plt.title('mean+/- 2std')
#
#plt.figure(10)
#plt.errorbar(x,kf2,yerr = kf2_var,fmt='o',label = 'KF2')
#plt.axhline(y=0.5,color='k',ls='dashed')
##plt.ylim((0,0.7))
#plt.xlim((0,2.2))
#plt.ylabel('stat_var2')
#plt.xlabel('$\Delta$')
#plt.title('mean+/- 2std')
