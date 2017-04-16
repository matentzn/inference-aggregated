#==============================================================================
#Lotka Volterra (LV) example
# Filtering plots for KF2
# uses the aggregated data with the inferred parameters
#==============================================================================
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from deriv_LV2 import deriv_LV2
from gil_integr_LV2 import gil_integr_LV2
import random 

random.seed(0)
np.random.seed(0)

n = 40#20#40
aggr_step = 2.0#5.0
m01 = 10
m02 = 100
c1 = 0.494#0.5
c2 = 0.0025#0.0025
c3 = 0.3#0.3

###### generate data ########################################################################
[obs_trapz,real_obs, obs_integr,time1,species] = gil_integr_LV2(m01,m02,n,aggr_step,c1,c2,c3)
obs = obs_trapz[:, 0]
s_noise = 3.0#2.0 #2.0
noise = np.matrix(np.random.normal(0, s_noise, len(obs))).T
obs = obs+noise # add noise in the aggregated data
real_obs1 = real_obs[:,0]
real_obs2 = real_obs[:,1]
#obs_avg = obs/aggr_step + noise
#obs_avg = real_obs1+noise

P = np.matrix([1,0])
V = np.matrix([s_noise**2])
m0 = np.matrix([m01,m02])
S0 = np.matrix([[0.1,0],[0,0.1]])
P0 = np.matrix([[1,0],[0,1]])
V0 = np.matrix([[0,0],[0,0]])

c = np.array([0.5,0.0025,0.3])

mean0 = P0*m0.T
cov0 = P0*S0*P0.T+V0
y0 = np.random.multivariate_normal(np.array(mean0.T)[0],np.array(cov0))
y0 = np.matrix(y0)

part1_0 = S0*P0.T*(P0*S0*P0.T+V0).I
m0_star = m0.T + part1_0*(y0.T-P0*m0.T)
S0_star = S0 - part1_0*P0*S0

init = np.array([m0_star[(0,0)],m0_star[(1,0)],S0_star[(0,0)],S0_star[(0,1)],S0_star[(1,1)],0*m0_star[(0,0)],0*m0_star[(1,0)],0*S0_star[(0,0)],0*S0_star[(0,1)],0*S0_star[(0,1)],0*S0_star[(1,1)],0*S0_star[(0,0)],0*S0_star[(0,1)],0*S0_star[(1,1)]])
y_det = np.array([100,100])
y_all = np.array([100,100,0.1,0,0.1,0,0,0,0,0,0,0,0,0])
t_all = np.array([0])

j = 0
step = aggr_step
dt = 0.001
mx_star_mat = np.array([0,0])
S_star_mat = np.array([[0,1],[1,0]])
for i in np.arange(0,n,step):
    time = np.arange(i,i+step+dt,dt)
    y,info = odeint(deriv_LV2,init,time,args = (c,),full_output = True)#,rtol=1e-3,atol=1e-3)#,mxstep=500,mxords=1,rtol=1e-4)

    l = len(y)
    
    t_all = np.concatenate((t_all,time))
    y_det = np.vstack((y_det,y[:,0:2]))
    y_all = np.vstack((y_all,y))
    
    mx = np.matrix([y[l-1, 0], y[l-1, 1]])
    S = np.matrix([[y[l-1, 2], y[l-1, 3]], [y[l-1, 3], y[l-1, 4]]])
    
    m = np.matrix([y[l-1, 5], y[l-1, 6]])
    Q = np.matrix([[y[l-1, 11], y[l-1, 12]], [y[l-1, 12], y[l-1, 13]]])
    QM = np.matrix([[y[l-1, 7], y[l-1, 8]], [y[l-1, 9], y[l-1, 10]]])

    part1x = QM.T*P.T*(P*Q*P.T+V).I
    mx_star = mx.T + part1x*(obs[j].T-P*m.T)
    mx_star_mat = np.vstack((mx_star_mat,np.array(mx_star).T))
    
    S_star = S - part1x*P*QM
    S_star_mat = np.vstack((S_star_mat,np.array(S_star)))
    
    init = np.array([mx_star[(0,0)],mx_star[(1,0)],S_star[(0,0)],S_star[(0,1)],S_star[(1,1)],0*mx_star[(0,0)],0*mx_star[(1,0)],0*S_star[(0,0)],0*S_star[(0,1)],0*S_star[(0,1)],0*S_star[(1,1)],0*S_star[(0,0)],0*S_star[(0,1)],0*S_star[(1,1)]])

    j +=1

mean1 = mx_star_mat[1:len(mx_star_mat),0]
mean2 = mx_star_mat[1:len(mx_star_mat),1]

variance1 = S_star_mat[2:len(S_star_mat):2,0]
variance2 = S_star_mat[3:len(S_star_mat):2,1]


xronos = np.arange(step,n+step,step)

plt.figure(1)
plt.plot(time1,species[:,0],'k')
plt.plot(t_all,y_det[:,0])
plt.xlim((0,n))
    
plt.figure(2)
plt.plot(time1,species[:,1],'k')
plt.plot(t_all,y_det[:,1])
plt.xlim((0,n))

plt.figure(1)
plt.plot(xronos,obs,'ro')
plt.plot(t_all,y_all[:,0]-np.sqrt(y_all[:,2]),'g')
plt.plot(t_all,y_all[:,0]+np.sqrt(y_all[:,2]),'g')
plt.plot(t_all,y_all[:,0],'magenta')
plt.xlabel('time')
plt.ylabel('observed species-prey')
plt.title('KF2-Restarting')
plt.xlim((0,n))

plt.figure(2)
#plt.plot(xronos,real_obs2,'ro')
plt.plot(t_all,y_all[:,1]-np.sqrt(y_all[:,4]),'g')
plt.plot(t_all,y_all[:,1]+np.sqrt(y_all[:,4]),'g')
plt.plot(t_all,y_all[:,1],'magenta')
plt.xlim((0,n))
plt.xlabel('time')
plt.ylabel('unobserved species-predator')
plt.title('KF2-Restarting')

plt.figure(5)
plt.errorbar(xronos,mean1,yerr=2*np.sqrt(variance1),fmt=None,color='b')
plt.plot(xronos,real_obs1,'ro')

plt.title('KF2 aggregate - observed')
plt.xlim((0,n+step))

plt.figure(6)
plt.errorbar(xronos,mean2,yerr=2*np.sqrt(variance2),fmt=None,color='b')
plt.plot(xronos,real_obs2,'ro')

plt.title('KF2 aggregate - unobserved')
plt.xlim((0,n+step))
plt.show()

plt.figure(101)
plt.plot(time1,species[:,0],'k')
plt.plot(t_all,y_det[:,0])
plt.xlim((0,n))
plt.xlabel('time')
plt.ylabel('observed species-prey')
plt.title('Macroscopic solution aggregate- Restaring')
    
plt.figure(102)
plt.plot(time1,species[:,1],'k')
plt.plot(t_all,y_det[:,1])
plt.xlim((0,n))
plt.xlabel('time')
plt.ylabel('unobserved species-predator')
plt.title('Macroscopic solution aggregate- Restaring')

