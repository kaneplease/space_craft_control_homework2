# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:59:20 2017

@author: 篤樹
"""

import numpy as np
import matplotlib.pyplot as plt

Ixx = 1.9
Iyy = 1.6
Izz = 2.0

#ステップ数
n = 2000
dt = 0.01

Mx = np.array([0.0]*(n+1))
My = np.array([0.0]*(n+1))
Mz = np.array([0.0]*(n+1))

omega_s = [0.1,1.88,0.0]
q_s     = [1.0,0.0,0.0,0.0]

def simulator(Mx,My,Mz,omega_s,q_s,n):    
    omega_list_x = []
    omega_list_y = []
    omega_list_z = []
    q0_list = []    
    q1_list = []
    q2_list = []
    q3_list = []    
    
    time = [dt]
    
    #最初の値
    omega_x,omega_y,omega_z = runge_kutta_omega(omega_s[0],omega_s[1],omega_s[2],\
                                            Mx[-1],My[-1],Mz[-1])
    q0,q1,q2,q3 = runge_kutta_q(q_s[0],q_s[1],q_s[2],q_s[3],omega_s[0],\
                                    omega_s[1],omega_s[2],Mx[-1],My[-1],Mz[-1])
    
    omega_list_x.append(omega_x)
    omega_list_y.append(omega_y)
    omega_list_z.append(omega_z)
    q0_list.append(q0)
    q1_list.append(q1)
    q2_list.append(q2)
    q3_list.append(q3)
    
    for i in range(n):
        omega_x,omega_y,omega_z = runge_kutta_omega(omega_list_x[i],omega_list_y[i],\
                                              omega_list_z[i],Mx[i],My[i],Mz[i])
        q0,q1,q2,q3 = runge_kutta_q(q0_list[i],q1_list[i],q2_list[i],q3_list[i],\
                                    omega_list_x[i],omega_list_y[i],\
                                              omega_list_z[i],Mx[i],My[i],Mz[i])
        
        #print q0**2+q1**2+q2**2+q3**2
        omega_list_x.append(omega_x)
        omega_list_y.append(omega_y)
        omega_list_z.append(omega_z)
        q0_list.append(q0)
        q1_list.append(q1)
        q2_list.append(q2)
        q3_list.append(q3)
        
        time.append((i+1)*dt)
      
    
    #返り値を見やすくするために一旦まとめる
    omega_list = [omega_list_x,omega_list_y,omega_list_z]
    q_list     = [q0_list,q1_list,q2_list,q3_list] 
    
    #プロット
    '''
    plt.plot(time,omega_list_x)
    plt.plot(time,omega_list_y)
    '''    
    plt.plot(time,omega_list_z)
    plt.show()
    '''
    plt.plot(time,q0_list)
    plt.plot(time,q1_list)
    plt.plot(time,q2_list)
    plt.plot(time,q3_list)
    plt.show()
    '''
    #print q_list,omega_list
    return q_list,omega_list,time
    
    
    
def d_q(q0,q1,q2,q3,omega_x,omega_y,omega_z,Mx,My,Mz):
    omega_now= np.array([omega_x,omega_y,omega_z])
    q_matrix = np.array([[-q1,-q2,-q3],\
                         [ q0,-q3, q2],\
                         [ q3, q0,-q1],\
                         [-q2, q1, q0]])
                         
    q_dot = 0.5*np.dot(q_matrix,omega_now)
    return q_dot[0],q_dot[1],q_dot[2],q_dot[3]
    

def d_omega(omega_x,omega_y,omega_z,Mx,My,Mz):
    omega_x_dot = Mx/Ixx - (Izz-Iyy)/Ixx*omega_y*omega_z
    omega_y_dot = My/Iyy - (Ixx-Izz)/Iyy*omega_x*omega_z
    omega_z_dot = Mz/Izz - (Iyy-Ixx)/Izz*omega_x*omega_y
    return omega_x_dot,omega_y_dot,omega_z_dot
    
def runge_kutta_omega(omega_x,omega_y,omega_z,Mx,My,Mz):
    #x,y,zぞれぞれの要素をふくんだリスト
    k1 = [0.0]*3
    k2 = [0.0]*3
    k3 = [0.0]*3
    k4 = [0.0]*3
    
    k1[0],k1[1],k1[2] = d_omega(omega_x,omega_y,omega_z,Mx,My,Mz)
    
    k2[0],k2[1],k2[2] = d_omega(omega_x + dt/2.0*k1[0],omega_y + dt/2.0*k1[1],\
                                omega_z + dt/2.0*k1[2],Mx,My,Mz)
    k3[0],k3[1],k3[2] = d_omega(omega_x + dt/2.0*k2[0],omega_y + dt/2.0*k2[1],\
                                omega_z + dt/2.0*k2[2],Mx,My,Mz)
    k4[0],k4[1],k4[2] = d_omega(omega_x + dt*k3[0],omega_y + dt*k3[1],\
                                omega_z + dt*k3[2],Mx,My,Mz)
    
    omega_now  = [omega_x,omega_y,omega_z]
    omega_next = [0.0]*3
    omega_next[0] = omega_now[0] + dt/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
    omega_next[1] = omega_now[1] + dt/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
    omega_next[2] = omega_now[2] + dt/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])    
    
    
    return omega_next[0],omega_next[1],omega_next[2]
    
def runge_kutta_q(q0,q1,q2,q3,omega_x,omega_y,omega_z,Mx,My,Mz):
    k1 = [0.0]*4
    k2 = [0.0]*4
    k3 = [0.0]*4
    k4 = [0.0]*4
    
    k1[0],k1[1],k1[2],k1[3] = d_q(q0,q1,q2,q3,omega_x,omega_y,omega_z,Mx,My,Mz)
    
    k2[0],k2[1],k2[2],k2[3] = d_q(q0+ dt/2.0*k1[0], q1+ dt/2.0*k1[1],\
                                q2+ dt/2.0*k1[2] ,q3+ dt/2.0*k1[3],\
                                omega_x,omega_y,omega_z,Mx,My,Mz)
                                
    k3[0],k3[1],k3[2],k3[3] = d_q(q0+ dt/2.0*k2[0], q1+ dt/2.0*k2[1],\
                                q2+ dt/2.0*k2[2], q3+ dt/2.0*k2[3],\
                                omega_x,omega_y,omega_z,Mx,My,Mz)
                                
    k4[0],k4[1],k4[2],k4[3] = d_q(q0+ dt*k3[0], q1+ dt*k3[1],\
                                q2+ dt*k3[2], q3+ dt*k3[3],\
                                omega_x,omega_y,omega_z,Mx,My,Mz)
    
    q0_next = q0 + dt/6.0 * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
    q1_next = q1 + dt/6.0 * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
    q2_next = q2 + dt/6.0 * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
    q3_next = q3 + dt/6.0 * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3])
    
    return q0_next,q1_next,q2_next,q3_next
    
if __name__ == '__main__':
    simulator(Mx,My,Mz,omega_s,q_s,n)
    
    
    
    
    