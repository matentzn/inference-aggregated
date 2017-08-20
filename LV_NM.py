import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from deriv_LV import deriv_LV
from LV_jac import LV_jac
from LV2_jac import LV2_jac
from deriv_LV2 import deriv_LV2
from gil_integr_LV2 import gil_integr_LV2
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import random

limit = 100
epanalipsi = 0
c_optim1_mat = np.zeros((limit,3))
c_optim2_mat = np.zeros((limit,3))
while epanalipsi <limit:
    print "epanalipsi"+str(epanalipsi)
    random.seed(epanalipsi)#1000#998
    np.random.seed(epanalipsi)#1000#998

    n = 20 
    length_of_interval = n
    aggr_step =  2.0
    m01 = 10#40#10
    m02 = 100#140#100
    c1 = 0.5
    c2 = 0.0025
    c3 = 0.3    
    print "aggr_step"+str(aggr_step)
    num_rep = 40#20
    real_obs1 = np.zeros((num_rep,(n/aggr_step)))
    real_obs2 = np.zeros((num_rep,(n/aggr_step)))
    obs = np.zeros((num_rep,(n/aggr_step)))
    obs_avg = np.zeros((num_rep,(n/aggr_step)))
    s_noise = 3.0#2.0 #2.0
    for rep in range(0,num_rep):
        [obs_trapz,real_obs, obs_integr,time1,species] = gil_integr_LV2(m01,m02,n,aggr_step,c1,c2,c3)
        
        obs1 = obs_trapz[:,0]
        real1 = real_obs[:,0]
        real2 = real_obs[:,1]
        #obs = real_obs1
        noise = np.matrix(np.random.normal(0,s_noise,len(obs1))).T
        obs_avg1 = obs1/aggr_step+noise
        obs1 = obs1+noise
        real_obs1[rep,:] = real1.T
        real_obs2[rep,:] = real2.T
        obs[rep,:] = obs1.T
        obs_avg[rep,:] = obs_avg1.T
    
    plt.figure(3)
    plt.plot(obs1/aggr_step,'ro')
    plt.plot(obs_integr[:,0],'ko')
    plt.plot(obs1[:,0],'bo')
    plt.plot(real1[:,0],'go')
    plt.xlim((0,n/aggr_step+0.1))
    
    
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
                #print m_star
                #print S_star
                init = np.array([m_star[(0,0)],m_star[(1,0)],S_star[(0,0)],S_star[(0,1)],S_star[(1,1)]])    
                j +=1
            sum_all = sum_all + suml
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
        
    c0 = np.array([0.5,0.0025,0.3])  
    #c0 = np.array([random.uniform(0.1,1),random.uniform(0,0.01),random.uniform(0.1,1)])    
    print c0
    c0 = np.log(c0)
    
    print "log:  "+str(c0)
    optim1 = minimize(neg_likelihood1,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
    print optim1
    print "optim1:  "+str(np.exp(optim1.x))
    
    def neg_likelihood2(c):
        return -likelihood2(c)

    print c0
    optim2 = minimize(neg_likelihood2,c0,method='Nelder-Mead',options={'disp': False, 'maxfev': 3000, 'maxiter': 3000})
    print optim2
    print "optim2:  "+str(np.exp(optim2.x))
    c_optim1_mat[epanalipsi,:] = np.exp(optim1.x)
    c_optim2_mat[epanalipsi,:] = np.exp(optim2.x)

    epanalipsi += 1

#c_optim1_mat.dump("c1_LV100_40reps_NM.dat")
#c_optim2_mat.dump("c2_LV100_40reps_NM.dat")

print np.mean(c_optim1_mat,axis=0)
print np.std(c_optim1_mat,axis=0)
print np.percentile(c_optim1_mat,50,axis=0)
print np.percentile(c_optim1_mat,75,axis=0)
print np.percentile(c_optim1_mat,25,axis=0)

print np.mean(c_optim2_mat,axis=0)
print np.std(c_optim2_mat,axis=0)
print np.percentile(c_optim2_mat,50,axis=0)
print np.percentile(c_optim2_mat,75,axis=0)
print np.percentile(c_optim2_mat,25,axis=0)

#with open("c1_optim_lv.bin", "wb") as output1:
#    pickle.dump(c_optim1_mat, output1)
#    
#with open("c2_optim_lv-expon.bin", "wb") as output2:
#    pickle.dump(c_optim1_mat, output2)    