#==============================================================================
# Creating synthetic data for LV using the Gillespie algorithm
# input
# x0_1 = initial abundance of prey population
# x0_1 = initial abundance of predator population
# aggr_step = aggregation period
# c1 = prey production rate
# c2 = predator production rate
# c3= predator death rate
# output smat, real, obs_integr,time,species
# smat = matrix of aggregated observations using the trapezoidal rule
# real = matrix of the underlying value of the observation (sampling from the unintegrated process)
# obs_integr = matrix of aggregated observations using the update formula dY_t = X_tdt
# time = matrix of the reaction times
# species = matrix of the updated states for each reaction time
#==============================================================================
from gillespieLV import GillespieLV
import numpy as np
import matplotlib.pyplot as plt

def gil_integr_LV2(x0_1,x0_2,m,aggr_step,c1,c2,c3):
    S = np.array([[1,0],[-1,1],[0,-1]])

    for j in range(0,1):
        x = np.array([x0_1,x0_2])
        gil = GillespieLV(S,m,x,c1,c2,c3)
        pinakes = gil.calc_gillespie()
        species = pinakes[0]
        time = pinakes[1]
        n = time.size
    plt.figure(1)
    plt.plot(time,species[:,0],'k')
    plt.xlim((0,m))

    plt.figure(2)   
    plt.plot(time,species[:,1],'k')
    plt.xlim((0,m))
    
    #Approximate updating of order1
    aggr1 = np.zeros(n)
    aggr2 = np.zeros(n)
    aggr1[0] = species[0, 0]
    aggr2[0] = species[0, 1]
    for i in range(0,n-1):
        aggr1[i+1] = aggr1[i] + species[i,0]*(time[i+1]-time[i])
        aggr2[i+1] = aggr2[i] + species[i,1]*(time[i+1]-time[i])
    
    # functions for aggreagated observations using the update formula
    def aggregate_measurements(obs_all,t):
        y_aggr = np.zeros(len(obs_all))
        y_aggr[0] = 0#obs_all[0]
        for i in range(0,len(obs_all)-1):
            #print i
            y_aggr[i+1] = y_aggr[i] + obs_all[i]*(t[i+1]-t[i])
        return y_aggr
    
    # aggregated observations (trapezoidal rule)
    k = 1
    #j1 = 1
    obs = 0
    real = np.array([0,0])
    xronos = 0
    #aggr_step = 0.5
    smat = np.array([0,0])
#    integr = np.array([0,0])
    obs_integr = np.array([0,0])
    #m = 25
    keep = 0
    for i in range(0,n):
            if time[i] > k*1*aggr_step: #and j1 < float(1)/aggr_step*m+1:  #collect discrete time points closest to 1, 2, 3,...
                    #print species[i,:],time[i], i
                    y1 = species[keep:i+1,0]
                    y2 = species[keep:i+1,1]
                    t = time[keep:i+1]
                    keep = i+1
                    #print keep
                    #print len(y1),y2,len(t)
                    y1_aggr = np.trapz(y1,t[:,0]) 
                    y2_aggr = np.trapz(y2,t[:,0]) 
                    obs = np.array([y1_aggr,y2_aggr])
                    # integrate using updating formulas
                    y_aggr_1 = aggregate_measurements(y1,t[:,0]) 
                    y_aggr_2 = aggregate_measurements(y2,t[:,0]) 
                    y_aggr_all = np.array([y_aggr_1[len(y_aggr_1)-1],y_aggr_2[len(y_aggr_2)-1]])
                    
                    k += 1
                    
                    smat = np.vstack((smat,obs))
                    obs_integr = np.vstack((obs_integr,y_aggr_all))
                    real = np.vstack((real,species[i,:]))
                    xronos = np.vstack((xronos,time[i]))
                    obs = 0
                    
                    
    smat = np.delete(smat,(0),axis = 0)   
    smat = np.asmatrix(smat)
    real = np.delete(real,(0),axis = 0)
    real = np.asmatrix(real)
    #integr = np.delete(integr,(0),axis = 0)  
    #integr = np.asmatrix(integr)
    xronos = np.delete(xronos,(0),axis = 0)  
    xronos = np.asmatrix(xronos)
    #print xronos
    obs_integr = np.delete(obs_integr,(0),axis = 0)  
    obs_integr = np.asmatrix(obs_integr)
        
    return smat, real, obs_integr,time,species#integr

############### test  run ####################
## do not exceed too much, danger of 0 species 
#n = 40 #20
#length_of_interval = n
#aggr_step =  2.0
#m01 = 10#71
#m02 = 100#79
#print "aggr_step"+str(aggr_step)
#num_rep = 1
#c1 = 0.5
#c2 = 0.0025
#c3 = 0.3
#real_obs1 = np.zeros((num_rep,(n/aggr_step)))
#real_obs2 = np.zeros((num_rep,(n/aggr_step)))
#obs = np.zeros((num_rep,(n/aggr_step)))
#obs_avg = np.zeros((num_rep,(n/aggr_step)))
#s_noise = 2.0
#for rep in range(0,num_rep):
#    [obs_trapz,real_obs, obs_integr,time1,species] = gil_integr_LV2(m01,m02,n,aggr_step,c1,c2,c3)
#                
#    obs1 = obs_trapz[:,0]
#    print obs1
#    real1 = real_obs[:,0]
#    real2 = real_obs[:,1]
#    if len(real1)<n/aggr_step:
#        addit0 = np.zeros(n/aggr_step-len(real1))
#        real1 = np.append(np.squeeze(np.asarray(real1)),addit0)
#        real1 = np.matrix(real1).T
#        real2 = np.append(np.squeeze(np.asarray(real2)),addit0)
#        real2 = np.matrix(real2).T        
#        obs1[len(obs1)-1] = 0
#        obs1 = np.append(np.squeeze(np.asarray(obs1)),addit0)
#        obs1 = np.matrix(obs1).T
#
#    #obs = real_obs1
#    noise = np.matrix(np.random.normal(0,s_noise,len(obs1))).T
#    obs_avg1 = obs1/aggr_step+noise
#    obs1 = obs1+noise
#    real_obs1[rep,:] = real1.T
#    real_obs2[rep,:] = real2.T
#    obs[rep,:] = obs1.T
#    obs_avg[rep,:] = obs_avg1.T
#    plt.figure(5)
#    plt.plot(time1,species[:,0],label='prey')
#    #plt.xlim((0,n))
#    #plt.title('Gillespie-prey')
#    #plt.figure(6)
#    plt.plot(time1,species[:,1],label='predator')
#    plt.xlim((0,n))
#    plt.title('Lotka-Voltera')
#    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
#    #plt.ylim((-20,120))
