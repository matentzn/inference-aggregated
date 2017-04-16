#==============================================================================
#Translation inhibition example with luciferase data
#likelihood 2 gives likelihood using KF2 (integrated LNA)
#Runs adaptive MCMC with 3 chains
#==============================================================================
import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#from trans_inh_ode2 import trans_inh_ode2
from transl_inh_ode_anal import transl_inh_ode_anal

from MCMC_BW_adapted_roberts import MCMC_BW_adapted_roberts

#from scipy.stats import kde
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import uniform
#import emcee
#import triangle 
import random 
#import xlrd
from xlrd import open_workbook
from xlrd import XL_CELL_TEXT, XL_CELL_NUMBER, XL_CELL_DATE, XL_CELL_BOOLEAN
random.seed(1000)
np.random.seed(1000)

n = 8.5
length_of_interval = n
aggr_step = 0.5

num_rep = 12

def sheet_to_array(filename, sheet_number, first_col=0, last_col=None, header=True):
    """Return a floating-point numpy array from sheet in an Excel spreadsheet.

    Notes:
    0. The array is empty by default; and any non-numeric data in the sheet will
       be skipped.
    1. If first_col is 0 and last_col is None, then all columns will be used,
    2. If header is True, only one header row is assumed.
    3. All rows are loaded.
    """
    DEBUG = False
    # sheet
    book = open_workbook(filename)
    sheet0 = book.sheet_by_index(sheet_number)
    rows = sheet0.nrows
    # cols
    if not last_col:
        last_col = sheet0.ncols
    if first_col >= last_col:
        raise Exception("First column must be smaller than last column!")
    cols = [col for col in range(first_col, last_col + 1)]
    # rows
    skip = 0
    if header:
        skip = 1
    data = np.empty([len(cols), rows - skip])

    for row in range(skip, sheet0.nrows):
        row_values = sheet0.row(row)
        for col, cell in enumerate(row_values):
            if DEBUG and row < 2:
                print row, col, cell.ctype, cell.value, '\n'
            if col in cols and cell.ctype == XL_CELL_NUMBER:
                data[col - first_col, row - skip] = cell.value
    return data


obs = sheet_to_array('luci1_reduced.xlsx', 0, first_col=0, last_col=None, header=False)
####start from 0
#init_luci = sheet_to_array('luci1_reduced_initial.xlsx', 0, first_col=0, last_col=None, header=False)
#init_luci = init_luci[0:num_rep,0]
#obs = obs[0:num_rep,0:18]
#n = 9
#init_luci = np.mean(init_luci)
#######################################
####start from 1
init_luci = np.mean(obs[0:num_rep,0]/0.5)#/0.5
obs = obs[0:num_rep,1:18]
########################################
obs_avg = obs/0.5
init_luci_log = np.log(init_luci)

def likelihood1(c) :
    c = np.exp(c)
    c[0] = c[0]/c[3]    
    c[4] = c[4]/c[3]
    V = np.matrix([c[2]])
    m0 = np.matrix([c[4]])
    S0 = np.matrix([0.1])
    V0 = np.matrix([1.0])
    #P0 = np.matrix([float(c[2])])
    P0 = np.matrix([1.0])
    P = np.matrix([float(c[3])])    
    #P = np.matrix([1.0])    

    lik0 = 1
    llik0 = 0

    part1_0 = S0*P0.T*(P0*S0*P0.T+V0).I
    m0_star = m0.T
    S0_star = (S0 - part1_0*P0*S0)*0
  
    sum_all = 0
    for r in range(0,num_rep):
        init = np.matrix([[m0_star[(0,0)]],[S0_star[(0,0)]]])    
        #init = np.array([m0_star[(0,0)],S0_star[(0,0)],0,0,0])
        j = 0
        step = aggr_step
        #dt = 0.001
        prodl = lik0
        suml = llik0
        m_star_mat = np.array([0])
        S_star_mat = np.array([1])
        for i in np.arange(0, n, step):
            
            ############################## analytical solution##########################
            t0 = i
            #print t0
            t1 = i+step
            y = transl_inh_ode_anal(init,t0,t1,c[1],c[0])
                  
            m = np.matrix([y[0,0]])
            S = np.matrix([y[1,0]])    
           
        ########################################## numerical solution/ change init too #########################
#            time = np.arange(i, i+step+dt, dt)
#            y = odeint(trans_inh_ode2,init,time,args = (c,))
#            l = len(y)
#            mx = np.matrix([y[l-1,0]])
#            S = np.matrix([y[l-1,1]])    
#            m = np.matrix([y[l-1,2]])
#            Q = np.matrix([y[l-1,4]])
#            QM = np.matrix([y[l-1,3]])
       ###################################################################################################     
            mean1 = P*m.T
            cov1 = P*S*P.T+V
            try:
                multivariate_normal.pdf(obs_avg[r,j], mean=mean1, cov=cov1)
            except ValueError:
                print "y1:  "+str(y)
                print "c1:  "+str(c)
                print "value1:  "+str(obs[r,j])
                print "r1:  "+str(r)                
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
            init = np.matrix([[m_star[(0,0)]],[S_star[(0,0)]]])
            #init = np.array([mx_star[(0,0)],S_star[(0,0)],0,0,0])    
            j +=1
        sum_all = sum_all + suml
    return sum_all
    

def posterior1(c):
    exp_c = np.exp(c)
    #return likelihood1(c)+norm.logpdf(float(c[4]),init_luci_log,1.)+norm.logpdf(float(c[2]),0.,2.0)# +c[0]+c[1]+c[2]+c[3]#+c[0]+c[1]+c[2]-num_rep*c[3]+sum(c[4:4+num_rep]) 
    #return likelihood1(c) +c[0]+c[1]+c[2]-c[3]+c[4]    
    #return likelihood1(c)+norm.logpdf(float(c[4]),init_luci_log,1.)+norm.logpdf(float(c[2]),-4.0,2.0)# +c[0]+c[1]+c[2]+c[3]#+c[0]+c[1]+c[2]-num_rep*c[3]+sum(c[4:4+num_rep]) 
    return likelihood1(c)+ expon.logpdf(float(exp_c[0]/exp_c[3]),loc=0, scale=10000) +expon.logpdf(float(exp_c[1]),loc=0, scale=10000)+expon.logpdf(float(exp_c[2]),loc=0, scale=10000)+expon.logpdf(float(exp_c[3]),loc=0, scale=10000)+expon.logpdf(float(exp_c[4]/exp_c[3]),loc=0, scale=10000) +c[0]+c[1]+c[2]-c[3]+c[4]
   

############################################################### KF2 ####################################################
run = 100000#200000#15000#3000
burn_in = 60000#2000
lamda1 = 0.1
n_chains = 3
len_chain = run-burn_in
chain_param1 = np.zeros([n_chains,len_chain])
chain_param2 = np.zeros([n_chains,len_chain])
chain_param3 = np.zeros([n_chains,len_chain])
chain_param4 = np.zeros([n_chains,len_chain])
chain_param5 = np.zeros([n_chains,len_chain])
chain_lik2 = np.zeros([n_chains,run])

c_mat_all = []
for ichain in range(0,n_chains):
    #c0 = np.array([norm.rvs(5.,1.0),norm.rvs(1.,0.1),uniform.rvs(0.,1.0),uniform.rvs(0.,1.0),init_luci])
    c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])    
    
    c0 = np.log(c0)  
   
    print c0
    while True:
        try:
            [c1_mat,rate1,lik1_mat,lambda1] = MCMC_BW_adapted_roberts(c0,run,posterior1,lamda1)
        except ValueError:
            print 'Try new c0'  
            c0 = np.array([uniform.rvs(0,1),uniform.rvs(0.1,1),uniform.rvs(0,1),uniform.rvs(0,1),uniform.rvs(1,20)])    
            c0 = np.log(c0) 
            print c0
            continue
        break
    
    print "step sizee1:  "+str(lambda1)
    print "Rate1:  "+str(rate1)
    c_mat_all.append(c1_mat)
    #c0 = np.array([expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0),expon.rvs(0.,1.0)])
    #c0 = np.log(c0)  
    c1_mat = np.exp(c1_mat)
   
    c12_mat = c1_mat[:,0]/c1_mat[:,3]
    c22_mat = c1_mat[:,1]
    c32_mat = c1_mat[:,2]
    c42_mat = c1_mat[:,3]
    c52_mat = c1_mat[:,4]/c1_mat[:,3]
    lik12_mat = lik1_mat[:,0]
    plt.figure(157)
    plt.plot(lik12_mat)
   
    plt.figure(7)
    plt.plot(c12_mat)
    plt.figure(8)
    plt.plot(c22_mat)
    plt.figure(9)
    plt.plot(c32_mat)
    plt.figure(10)
    plt.plot(c42_mat)
    plt.figure(11)
    plt.plot(c52_mat)
   
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

plt.figure(17)
plt.hist(thinned21, 5,normed=True, histtype='step',color = 'g')
#plt.hist(cp_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[0],0,2.0,colors=u'r')
plt.xlabel(r'$cp$', fontsize=24)
plt.ylabel(r'posterior', fontsize=24)
plt.show()

plt.figure(18)
plt.hist(thinned22, 5,normed=True, histtype='step',color = 'g')
#plt.hist(dp_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[1],0,0.5,colors=u'r')
plt.xlabel(r'$dp$', fontsize=24)
plt.ylabel(r'posterior', fontsize=24)
plt.show()

plt.figure(19)
plt.hist(thinned23, 5,normed=True, histtype='step',color = 'g')
#plt.hist(noise_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[2],0,3.0,colors=u'r')
plt.xlabel(r'$k$', fontsize=24)
plt.ylabel(r'posterior', fontsize=24)
plt.show()

plt.figure(20)
plt.hist(thinned24, 5,normed=True, histtype='step',color = 'g')
#plt.hist(m0_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[3],0,0.5,colors=u'r')
plt.xlabel(r'$m0$', fontsize=24)
plt.ylabel(r'posterior', fontsize=24)
plt.show()

plt.figure(21)
plt.hist(thinned25, 5,normed=True, histtype='step',color = 'g')
#plt.hist(k_lik1, 5,normed=True, histtype='step',color = 'b')
#plt.vlines(c_real[4],0,0.5,colors=u'r')
plt.xlabel(r'$noise$', fontsize=24)
plt.ylabel(r'posterior', fontsize=24)
plt.show()

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

########################### save data ##################################
#import pickle
#
#with open("c1_3chains_luci_1init.bin", "wb") as output:
#    pickle.dump(c_mat_all, output)