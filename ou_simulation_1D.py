#==============================================================================
# ou_simulation_1D is used for 
# inputs
# x0:initial state for OU 
# n = length of sampling period
# step = aggregation period
# a = drift of OU
# sigma = diffusion constant of OU
# outputs 
# obs = aggregated observations (using trapezoidal rule)
# real_obs = non aggregated observations at aggregation times
# t = all times sampled 
# aggr2 = equivalent to obs using aggregate_measurements function
# ou = complete non aggregated trace of OU 
#==============================================================================
import numpy as np
import math
import random

def ou_simulation_1D(x0,n,step,a,sigma):
    dt = 0.0001
    t = np.arange(0,n+dt,dt)
    
    def euler_maruyama(tau,c,x0,t):
        
        tau = float(tau)
        c = float(c)
        print tau, c
        l = t.size
        w = np.zeros(l)
        w[0] = x0
                
        
        #sqrt_dt = math.sqrt(dt)
        for i in range(0,l-1):
            #### euler-maruyama ##################
            ##w[i+1] = w[i] -(1/tau)*w[i]*dt + np.sqrt(c)*random.gauss(0,1)*sqrt_dt
            #w[i+1] = w[i] -(tau)*w[i]*dt + c*sqrt_dt*np.random.normal(loc=0.0, scale=1.0) #random.gauss(0,1)
            # exact updating formula for w ########
            w[i+1] = w[i]*np.exp(-tau*dt)+ np.sqrt((c**2*0.5/tau)*(1-np.exp(-2*dt*tau)))*random.gauss(0,1)
        return w 
    
    ou = euler_maruyama(a,sigma,x0,t)

    def aggregate_measurements(obs_all,dt):
        y_aggr = np.zeros(len(obs_all)+1)
        y_aggr[0] = 0#obs_all[0]
        for i in range(0,len(obs_all)):
            y_aggr[i+1] = y_aggr[i] + obs_all[i]*dt
        return y_aggr
   
    real_obs = x0
    #aggr_obs_anal = x0
    aggr2 = x0
    obs = x0
    xronos = 0
    previous = 0
    #obs2 = x0
    for  i in range(0,int((1/step)*n)):
##        print t[i*(1/dt)], ou[i*(1/dt)]
        #print t[(float(i)*step)*(1/dt)], t[(float(i)*step)*(1/dt)+1], t[(float(i+1)*step)*(1/dt)]
        obs_all = ou[previous:(float(i+1)*step)*(1/dt)+1]  
        #time_all = t[(float(i+1)*step)*(1/dt)]
        #print time_all
        #print ou
        y_aggr_2 = aggregate_measurements(obs_all,dt)
        aggr2 = np.vstack((aggr2,y_aggr_2[len(y_aggr_2)-1]))         
        real_obs = np.vstack((real_obs,ou[(float(i+1)*step)*(1/dt)]))
        aggr_obs = dt*0.5*(ou[(float(i)*step)*(1/dt)]+2*(sum(ou[((float(i)*step)*(1/dt)+1):((float(i+1)*step)*(1/dt))]))+ou[((float(i+1)*step)*(1/dt))])
        obs = np.vstack((obs,aggr_obs))
        #aggr_obs2 = np.trapz(ou[(float(i)*step)*(1/dt):(float(i+1)*step)*(1/dt)],t[(float(i)*step)*(1/dt):(float(i+1)*step)*(1/dt)])                
        #obs2 = np.vstack((obs2,aggr_obs2))        
        #aggr_obs_anal = np.vstack((aggr_obs_anal,aggr[(float(i+1)*step)*(1/dt)]))
        xronos = np.vstack((xronos,t[(float(i+1)*step)*(1/dt)]))
        previous = (float(i+1)*step)*(1/dt)+1   
    
    return obs, real_obs, t, aggr2,ou
    
####### example plots ##############    
#import matplotlib.pyplot as plt
#n = 5
#x0 = 20
#step = 0.1 #give float number
#[obs,ou,time1,aggr_anal,trace] = ou_simulation_1D(x0,n,step,4.0,2.0)
#
#real_obs = ou[1:len(ou)]
#obs = aggr_anal[1:len(aggr_anal)]
##obs = real_obs
#obs_avg =  obs/step #real_obs #obs/step
#
#plt.figure(0)
##plt.plot(obs_anal,'ko')
#plt.plot(obs,'ro')
#plt.plot(real_obs,'go')
#plt.plot(obs/step,'ko')
#plt.xlim((0,5))
#
##plt.figure()
##plt.plot(obs,'ro')
##plt.plot(ou,'ko')
#plt.plot(time1,trace)
#plt.xlim((0,1))
#
##plt.plot(aggr_anal,'bo')
##plt.title('Ornstein Uhlenbeck')
##plt.xlim((0,5))
##plt.xlabel('time')
##plt.ylabel('X')
