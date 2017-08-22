#==============================================================================
#Lotka Volterra (LV) example
# Filtering plots for KF1
# uses the normalized data with the inferred parameters
#==============================================================================
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from deriv_LV import deriv_LV
from gil_integr_LV2 import gil_integr_LV2
import random 

random.seed(0)
np.random.seed(0)

n = 40#20
aggr_step = 2.0#5.0

m01 = 10
m02 = 100
c1 = 0.5#0.480253558024#0.5
c2 = 0.0025#0.00227124893865#0.0025
c3 = 0.3#0.242781264072#0.3

###### generate data ########################################################################
[obs_trapz,real_obs, obs_integr,time1,species] = gil_integr_LV2(m01,m02,n,aggr_step,c1,c2,c3)
obs = obs_trapz[:, 0]
s_noise = 3.0#2.0 #2.0
noise = np.matrix(np.random.normal(0, s_noise, len(obs))).T
obs = obs+noise
real_obs1 = real_obs[:,0]
real_obs2 = real_obs[:,1]
obs_avg = obs/aggr_step + noise # normalise aggregated data and add noise
#obs_avg = real_obs1+noise

P = np.matrix([1,0])
V = np.matrix([s_noise**2])
m0 = np.matrix([m01,m02])
S0 = np.matrix([[0.1,0],[0,0.1]])
P0 = np.matrix([[1,0],[0,1]])
V0 = np.matrix([[0,0],[0,0]])

#c = np.array([0.5,0.0025,0.3])
c = np.array([c1,c2,c3])

mean0 = P0*m0.T
cov0 = P0*S0*P0.T+V0
y0 = np.random.multivariate_normal(np.array(mean0.T)[0],np.array(cov0))
y0 = np.matrix(y0)

part1_0 = S0*P0.T*(P0*S0*P0.T+V0).I
m0_star = m0.T + part1_0*(y0.T-P0*m0.T)
S0_star = S0 - part1_0*P0*S0

init = np.array([m0_star[(0,0)],m0_star[(1,0)],S0_star[(0,0)],S0_star[(0,1)],S0_star[(1,1)]])
y_det = np.array([100,100])
y_all = np.array([100,100,0.1,0,0.1])
t_all = np.array([0])

j = 0
step = aggr_step
#x = np.zeros((9,2))
dt = 0.001
m_star_mat = np.array([0,0])
S_star_mat = np.array([[0,1],[1,0]])
for i in np.arange(0,n,step):
    time = np.arange(i,i+step+dt,dt)
    #y,info = odeint(deriv_LV,init,time,args = (c,))
    y,info = odeint(deriv_LV,init,time,args = (c,),full_output = True)#,rtol=1e-3,atol=1e-3)#,mxstep=500,mxords=1,rtol=1e-4)

    l = len(y)
    
    t_all = np.concatenate((t_all,time))
    y_det = np.vstack((y_det,y[:,0:2]))
    y_all = np.vstack((y_all,y))
    
    m = np.matrix([y[l-1,0],y[l-1,1]])
    S = np.matrix([[y[l-1,2],y[l-1,3]],[y[l-1,3],y[l-1,4]]])
    
    part1 = S*P.T*(P*S*P.T+V).I
    m_star = m.T + part1*(obs_avg[j].T-P*m.T)
    m_star_mat = np.vstack((m_star_mat,np.array(m_star).T))
    S_star = S - part1*P*S
    S_star_mat = np.vstack((S_star_mat,np.array(S_star)))
    #print m_star
    #print S_star
#    x[j,:] = np.random.multivariate_normal(m_star,S_star)
    init = np.array([m_star[(0,0)],m_star[(1,0)],S_star[(0,0)],S_star[(0,1)],S_star[(1,1)]])
    #init = np.array([m_star[(0,0)],m_star[(1,0)],0,0,0])

    j +=1

mean1 = m_star_mat[1:len(m_star_mat),0]
mean2 = m_star_mat[1:len(m_star_mat),1]

variance1 = S_star_mat[2:len(S_star_mat):2,0]
variance2 = S_star_mat[3:len(S_star_mat):2,1]


xronos = np.arange(step,n+step,step)
##plt.ion()
#plt.figure(3)
#plt.plot(xronos,mean1)
#plt.plot(xronos,mean1-np.sqrt(variance1),'g')
#plt.plot(xronos,mean1+np.sqrt(variance1),'g')
#plt.plot(xronos,real_obs1,'.')
#
#plt.figure(4)
#plt.plot(xronos,mean2)
#plt.plot(xronos,mean2-np.sqrt(variance2),'g')
#plt.plot(xronos,mean2+np.sqrt(variance2),'g')
#plt.plot(xronos,real_obs2,'.')
#plt.draw()


plt.figure(1)
plt.plot(time1,species[:,0],'k')
plt.plot(t_all,y_det[:,0])
plt.xlim((0,n))
    
plt.figure(2)
plt.plot(time1,species[:,1],'k')
plt.plot(t_all,y_det[:,1])
plt.xlim((0,n))

plt.figure(1)
plt.plot(xronos,obs_avg,'ro')
plt.plot(t_all,y_all[:,0]-np.sqrt(y_all[:,2]),'g')
plt.plot(t_all,y_all[:,0]+np.sqrt(y_all[:,2]),'g')
plt.plot(t_all,y_all[:,0],'magenta')
plt.xlabel('time')
plt.ylabel('observed species-prey')
plt.title('KF1')
plt.xlim((0,n))

plt.figure(2)
#plt.plot(xronos,real_obs2,'ro')
plt.plot(t_all,y_all[:,1]-np.sqrt(y_all[:,4]),'g')
plt.plot(t_all,y_all[:,1]+np.sqrt(y_all[:,4]),'g')
plt.plot(t_all,y_all[:,1],'magenta')
plt.xlim((0,n))
plt.xlabel('time')
plt.ylabel('unobserved species-predator')
plt.title('KF1')

plt.figure(5)
plt.errorbar(xronos,mean1,yerr=2*np.sqrt(variance1),fmt=None,color='b')
plt.plot(xronos,real_obs1,'ro')

plt.title('KF1 aggregate - observed')
plt.xlim((0,n+step))

plt.figure(6)
plt.errorbar(xronos,mean2,yerr=2*np.sqrt(variance2),fmt=None,color='b')
plt.plot(xronos,real_obs2,'ro')

plt.title('KF1 aggregate - unobserved')
plt.xlim((0,n+step))
plt.show()

plt.figure(101)
plt.plot(time1,species[:,0],'k')
plt.plot(t_all,y_det[:,0])
plt.xlim((0,n))
plt.xlabel('time')
plt.ylabel('observed species-prey')
plt.title('Macroscopic solution -Restaring')
    
plt.figure(102)
plt.plot(time1,species[:,1],'k')
plt.plot(t_all,y_det[:,1])
plt.xlim((0,n))
plt.xlabel('time')
plt.ylabel('unobserved species-predator')
plt.title('Macroscopic solution -Restaring')

