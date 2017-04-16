import numpy as np
import matplotlib.pyplot as plt
import pickle

#stat_var01 = np.load("/home/owl/Desktop/paper_code/stat_var1_all01.dat")
#stat_var05 = np.load("/home/owl/Desktop/paper_code/stat_var1_all05.dat")
#stat_var1 = np.load("/home/owl/Desktop/paper_code/stat_var1_all1.dat")
#stat_var2 = np.load("/home/owl/Desktop/paper_code/stat_var1_all2.dat")
#
#stat_var01_2 = np.load("/home/owl/Desktop/paper_code/stat_var2_all01.dat")
#stat_var05_2 = np.load("/home/owl/Desktop/paper_code/stat_var2_all05.dat")
#stat_var1_2 = np.load("/home/owl/Desktop/paper_code/stat_var2_all1.dat")
#stat_var2_2 = np.load("/home/owl/Desktop/paper_code/stat_var2_all2.dat")

with open("stat_var1_step01.bin", "rb") as data1:
   stat_var01 = pickle.load(data1)
with open("stat_var1_step05.bin", "rb") as data2:
   stat_var05 = pickle.load(data2)
with open("stat_var1_step1.bin", "rb") as data3:
   stat_var1 = pickle.load(data3)
with open("stat_var1_step2.bin", "rb") as data4:
   stat_var2 = pickle.load(data4)

with open("stat_var2_step01.bin", "rb") as data5:
   stat_var01_2 = pickle.load(data5)
with open("stat_var2_step05.bin", "rb") as data6:
   stat_var05_2 = pickle.load(data6)
with open("stat_var2_step1.bin", "rb") as data7:
   stat_var1_2 = pickle.load(data7)
with open("stat_var2_step2.bin", "rb") as data8:
   stat_var2_2 = pickle.load(data8)

stat_var01_mean = np.mean(stat_var01,axis=0)
stat_var05_mean = np.mean(stat_var05,axis=0)
stat_var1_mean = np.mean(stat_var1,axis=0)
stat_var2_mean = np.mean(stat_var2,axis=0)
#stat_var4_mean = np.mean(stat_var4,axis=0)

stat_var01_mean2 = np.mean(stat_var01_2,axis=0)
stat_var05_mean2 = np.mean(stat_var05_2,axis=0)
stat_var1_mean2 = np.mean(stat_var1_2,axis=0)
stat_var2_mean2 = np.mean(stat_var2_2,axis=0)
#stat_var4_mean2 = np.mean(stat_var4_2,axis=0)

stat_var01_std = np.std(stat_var01,axis=0)
stat_var05_std = np.std(stat_var05,axis=0)
stat_var1_std = np.std(stat_var1,axis=0)
stat_var2_std = np.std(stat_var2,axis=0)
#stat_var4_std = np.std(stat_var4,axis=0)

stat_var01_std2 = np.std(stat_var01_2,axis=0)
stat_var05_std2 = np.std(stat_var05_2,axis=0)
stat_var1_std2 = np.std(stat_var1_2,axis=0)
stat_var2_std2 = np.std(stat_var2_2,axis=0)
#stat_var4_std2 = np.std(stat_var4_2,axis=0)
plt.figure(1)
plt.boxplot([stat_var01,stat_var05,stat_var1,stat_var2], showfliers = False)#,whis="range")
plt.axhline(y=0.5,color='k',ls='dashed')  
plt.xticks([1, 2, 3, 4], [0.1, 0.5, 1.0, 2.0])  
plt.ylabel('stat_var', fontsize=16)
plt.xlabel('$\Delta$', fontsize=18)
plt.title('KF1', fontsize=16)

plt.figure(2)
plt.boxplot([stat_var01_2,stat_var05_2,stat_var1_2,stat_var2_2], showfliers = False)#,whis="range")
plt.axhline(y=0.5,color='k',ls='dashed')    
plt.xticks([1, 2, 3, 4], [0.1, 0.5, 1.0, 2.0])  
plt.ylabel('stat_var2', fontsize=16)
plt.xlabel('$\Delta$', fontsize=18)
plt.title('KF2', fontsize=16)

#for i in range(0,10):
#
#    #kf1 = np.array([stat_var001_mean[i],stat_var01_mean[i],stat_var05_mean[i],stat_var1_mean[i],stat_var2_mean[i]])
#    #kf1_var = 2*np.array([stat_var001_std[i],stat_var01_std[i],stat_var05_std[i],stat_var1_std[i],stat_var2_std[i]])
#    kf1 = np.array([stat_var01_mean[i],stat_var05_mean[i],stat_var1_mean[i],stat_var2_mean[i]])
#    kf1_var = 2*np.array([stat_var01_std[i],stat_var05_std[i],stat_var1_std[i],stat_var2_std[i]])
#    
#    #kf2 = np.array([stat_var001_mean2[i],stat_var01_mean2[i],stat_var05_mean2[i],stat_var1_mean2[i],stat_var2_mean2[i]])
#    #kf2_var = 2*np.array([stat_var001_std2[i],stat_var01_std2[i],stat_var05_std2[i],stat_var1_std2[i],stat_var2_std2[i]])    
#    kf2 = np.array([stat_var01_mean2[i],stat_var05_mean2[i],stat_var1_mean2[i],stat_var2_mean2[i]])
#    kf2_var = 2*np.array([stat_var01_std2[i],stat_var05_std2[i],stat_var1_std2[i],stat_var2_std2[i]])
#    
#    kf1_stat_var = np.array([stat_var01[i],stat_var05[i],stat_var1[i],stat_var2[i]])
#    kf2_stat_var = np.array([stat_var01_2[i],stat_var05_2[i],stat_var1_2[i],stat_var2_2[i]])
#
#    #x = np.array([0.01,0.1,0.5,1.0,2.0])
#    x = np.array([0.1,0.5,1.0,2.0])
#    
#    plt.figure(i)
#    plt.errorbar(x,kf1,yerr = kf1_var,fmt='o',label = 'KF1')
#    plt.axhline(y=0.5,color='k',ls='dashed')    
#    plt.ylim((0,1.0))
#    plt.xlim((0,2.2))
#    plt.ylabel('stat_var')
#    plt.xlabel('$\Delta$')
#    plt.title('mean+/- 2std')
#
#    plt.figure(i+10)
#    plt.errorbar(x,kf2,yerr = kf2_var,fmt='o',label = 'KF2')
#    plt.axhline(y=0.5,color='k',ls='dashed')
#    #plt.ylim((0,0.7))
#    plt.xlim((0,2.2))
#    plt.ylabel('stat_var2')
#    plt.xlabel('$\Delta$')
#    plt.title('mean+/- 2std')
#
#    plt.figure(i+20)
#    plt.boxplot([stat_var01[i],stat_var05[i],stat_var1[i],stat_var2[i]], showfliers = False)#,whis="range")
#    plt.axhline(y=0.5,color='k',ls='dashed')    
#    plt.ylabel('stat_var')
#    plt.xlabel('$\Delta$')
#
#    plt.figure(i+30)
#    plt.boxplot([stat_var01_2[i],stat_var05_2[i],stat_var1_2[i],stat_var2_2[i]], showfliers = False)#,whis="range")
#    plt.axhline(y=0.5,color='k',ls='dashed')    
#    plt.ylabel('stat_var2')
#    plt.xlabel('$\Delta$')
#    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
#    
