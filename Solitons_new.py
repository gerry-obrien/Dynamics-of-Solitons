# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:03:56 2022

@author: gerry
"""

import numpy as np
import matplotlib.pyplot as plt

T = 100
n_timesteps = 200
#dt = T / n_timesteps
dt = 0.0001

alpha = 0.5 # alpha for runge-kutta
h = 0.1

ALPHA = 5 # alpha for u0 calculation

u_start = -10
u_end = 10

numerator = (12 * ALPHA ** 2)
x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h))
u0 = numerator / (np.cosh(ALPHA * x_vals) ** 2)


def f(u,h):
    f_list = []
    for i in range(len(u)):
        # Calculate wrap around index for u_i
        i_p1 = (i + 1) % len(u)
        i_p2 = (i + 2) % len(u)
        i_m1 = (i - 1) % len(u)
        i_m2 = (i - 2) % len(u)
        
        t1 = -(1/(4*h)) * ((u[i_p1]**2) - (u[i_m1]**2))
        t2 = -(1/(2*h**3)) * (u[i_p2] - 2*u[i_p1] + 2*u[i_m1] - u[i_m2])
                         
        f = t1 + t2
        f_list.append(f)
    return np.array(f_list)

def sol(xs, t, ALPHA): # analytical solution
    return 12 * (ALPHA ** 2) / (np.cosh(ALPHA * (xs - (4 * ALPHA ** 2) * t)))

def fa(u, h):
    return f(u, h)

# calculate fb given un
def fb(u, fa, dt, h):
    return f(u + 0.5 * dt * fa, h)

def fc(u, fb, dt, h):
    return f(u + 0.5 * dt * fb, h)

def fd(u, fc, dt, h):
    return f(u + dt * fc, h)


#un = u0 # starting approximation
rows = [u0]  # list of all calculated un approximations
for i in range(n_timesteps):
    un = rows[-1]
    fa_n = fa(un, h)
    fb_n = fb(un, fa_n, dt, h)
    fc_n = fc(un, fb_n, dt, h)
    fd_n = fd(un, fc_n, dt, h)
    
    rows.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)
#%%
#def sol_rk_4(n)
#diff_list = []
for i in range(len(rows)):
    plt.plot(x_vals,rows[i])
    plt.plot(x_vals, sol(x_vals, i * dt, ALPHA))
    plt.show()
#    diff = sum(abs(sol(x_vals, i * dt, ALPHA) - rows[i]))
#    diff_list.append(diff)
#%%
# Plot to show numerical and analytical solutions at t=0s and t=0.2s
plt.plot(x_vals,rows[0],label='Numerical at t=0s')
plt.plot(x_vals,sol(x_vals,0,ALPHA),'--',color='black',label='Analytical at t=0s')
plt.plot(x_vals,rows[-1], label='Numerical at t=0.2s')
plt.plot(x_vals,sol(x_vals,0.0185,ALPHA),'--',color='black',label='Analytical at t=0.2s')
plt.legend()
plt.xlabel('x / m')
plt.ylabel('u / m')
#plt.xlim(-5,7.5)
plt.show()
#%%
#Calculating the number of timesteps before instability for different alpha and dt
#leave h as 0.1
#tested over 20 timesteps

ALPHA_list = [1,2,3,4,5,6,7,8,9,10]
dt_list = [0.1,0.01,0.001,0.0001]

num_tsteps = np.linspace(0,n_timesteps-1,n_timesteps)

x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h))

for i in ALPHA_list:
    for j in dt_list:
        numerator_test = (12 * i ** 2)
        u0_test = numerator_test / (np.cosh(i * x_vals) ** 2)
        rows_test = [u0_test]
        for k in range(n_timesteps - 1):
            un_test = rows_test[-1]
            fa_n_test = fa(un_test, h)
            fb_n_test = fb(un_test, fa_n_test, j, h)
            fc_n_test = fc(un_test, fb_n_test, j, h)
            fd_n_test = fd(un_test, fc_n_test, j, h)
            
            rows_test.append(un_test + (1/6) * (fa_n_test + 2 * fb_n_test + 2 * fc_n_test + fd_n_test) * j)
        
        diff_list = []
        
        for l in range(len(rows_test)):
            diff = sum(abs(sol(x_vals, l * j, i) - rows_test[l]))
            diff_list.append(diff)
        

        plt.plot(num_tsteps, diff_list, label = 'dt = %s s'%j)
        plt.title('Stability plot, alpha = %s'%i)
        plt.xlabel('Number of timesteps')
        plt.ylabel('Difference between analytic and numerical solutions')
        plt.xlim(0,20)
        plt.ylim(0,3000)
    plt.legend()
    plt.show()

#%%
#now for h=0.01
h_new = 0.01
x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h_new))

for i in ALPHA_list:
    for j in dt_list:
        numerator_test = (12 * i ** 2)
        u0_test = numerator_test / (np.cosh(i * x_vals) ** 2)
        rows_test = [u0_test]
        for k in range(n_timesteps - 1):
            un_test = rows_test[-1]
            fa_n_test = fa(un_test, h_new)
            fb_n_test = fb(un_test, fa_n_test, j, h_new)
            fc_n_test = fc(un_test, fb_n_test, j, h_new)
            fd_n_test = fd(un_test, fc_n_test, j, h_new)
            
            rows_test.append(un_test + (1/6) * (fa_n_test + 2 * fb_n_test + 2 * fc_n_test + fd_n_test) * j)
        
        diff_list = []
        
        for l in range(len(rows_test)):
            diff = sum(abs(sol(x_vals, l * j, i) - rows_test[l]))
            diff_list.append(diff)
        

        plt.plot(num_tsteps, diff_list, label = 'dt = %s'%j)
        plt.title('Stability plot, alpha = %s'%i)
        plt.xlabel('Number of timesteps')
        plt.ylabel('Difference between analytic and numerical solutions')
        plt.xlim(0,20)
#        plt.ylim(0,3000)
    plt.legend()
    plt.show()
#This shows that h=0.01 gives instabality for a large range of dt
#Hence h=0.1 is chosen

#%%
#Trapezoidal rule integration

def integrate(row, h):
    I_list = [0.5 * abs(row[0])]
    for i in range(1,len(row)-1):
        I_list.append(abs(row[i]))
    I_list.append(0.5 * abs(row[-1]))
    I = h * sum(I_list)
    return I

#%%
#two solitons verydifferent speeds
#making two solitons as an initial condition
ALPHA1 = 5
ALPHA2 = 2

dt=0.001

numerator1 = (12 * ALPHA1 ** 2)
numerator2 = (12 * ALPHA2 ** 2)

u0_1 = numerator1 / (np.cosh(ALPHA1 * x_vals) ** 2)
u0_2 = numerator2 / (np.cosh(ALPHA2 * (x_vals - 3))**2) #starts the slow pulse at x=3

u0_two = u0_1 + u0_2
#plt.plot(x_vals, u0_two)

n_timesteps = 60

rows_two = [u0_two]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows_two[-1]
    fa_n = fa(un, h)
    fb_n = fb(un, fa_n, dt, h)
    fc_n = fc(un, fb_n, dt, h)
    fd_n = fd(un, fc_n, dt, h)
    
    rows_two.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)

speeds = []
amplitudes = []
times = []
integrals = []

for i in range(len(rows_two)): #plotting the function at successive time steps
    plt.plot(x_vals,rows_two[i])
    plt.ylim(0,300)
    plt.show()
    
    time = i * dt #calculating a plotting amplitude and speed vs time
    times.append(time)
    amplitude = max(rows_two[i])
    amplitudes.append(amplitude)
    if i >= 1:
        row = rows_two[i].tolist()
        row1 = rows_two[i-1].tolist()
        max_x_val_i = (row.index(max(rows_two[i])) * h) + u_start
        max_x_val_i1 = (row1.index(max(rows_two[i-1])) * h) + u_start
        speed = (max_x_val_i - max_x_val_i1) / dt
    else:
        speed = 0
    speeds.append(speed)
    
    I = integrate(rows_two[i],h) #Calculating the integral of the graph at each timestep
    integrals.append(I)
    
    print('time = %s, amplitude = %s, speed = %s, Integral = %s' %(time,amplitude,speed,I)) # printing amplitudes and speeds
    
    
    
#plotting amplitude of highest peak vs time and speed of highest peak vs time
plt.plot(times, amplitudes)
#plt.title('Maximum amplitude vs Time')
plt.xlabel('Time / s')
plt.ylabel('Amplitude / m')
plt.show()

plt.plot(times, speeds)
plt.title('Speed of the peak vs Time')
plt.xlabel('Time \ s')
plt.ylabel('Speed of the peak / m/s')
plt.show()

plt.plot(times, integrals) # you see a slight increase in the integral due to the oscillations behind x=0 increasing
#plt.title('Integral vs Time') # there is no sign of the collision changing the integral suggesting area under the graph remains unchanged
plt.xlabel('Time / s')
plt.ylabel('Integral')
plt.show()
# amplitude of the taller soliton drops by the amplitide of the smaller soliton
#%%
#Plotting the two different speed soliton interaction
plt.plot(x_vals,rows_two[0], label='t=0s')
plt.plot(x_vals,rows_two[34], label='t=0.034s')
plt.plot(x_vals,rows_two[-1],label='t=0.06s')
plt.xlabel('x / m')
plt.ylabel('u / m')
plt.legend()
plt.xlim(-5,10)
plt.show()
#plt.plot(x_vals,rows_two[16])
#%%
#two solitons similar speeds
u_start = -10
u_end = 20

x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h))

ALPHA1 = 5
ALPHA2 = 4

numerator1 = (12 * ALPHA1 ** 2)
numerator2 = (12 * ALPHA2 ** 2)

u0_1 = numerator1 / (np.cosh(ALPHA1 * x_vals) ** 2)
u0_2 = numerator2 / (np.cosh(ALPHA2 * (x_vals - 2.5))**2) #starts the slow pulse at x=3

u0_two = u0_1 + u0_2
#plt.plot(x_vals, u0_two)

n_timesteps = 100

rows_two = [u0_two]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows_two[-1]
    fa_n = fa(un, h)
    fb_n = fb(un, fa_n, dt, h)
    fc_n = fc(un, fb_n, dt, h)
    fd_n = fd(un, fc_n, dt, h)
    
    rows_two.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)

speeds = []
amplitudes = []
times = []
integrals = []

for i in range(len(rows_two)): #plotting the function at successive time steps
    plt.plot(x_vals,rows_two[i])
    plt.ylim(0,300)
    plt.xlim(-1,20)
    plt.show()
    
    time = i * dt #calculating a plotting amplitude and speed vs time
    times.append(time)
    amplitude = max(rows_two[i])
    amplitudes.append(amplitude)
    if i >= 1:
        row = rows_two[i].tolist()
        row1 = rows_two[i-1].tolist()
        max_x_val_i = (row.index(max(rows_two[i])) * h) + u_start
        max_x_val_i1 = (row1.index(max(rows_two[i-1])) * h) + u_start
        speed = (max_x_val_i - max_x_val_i1) / dt
    else:
        speed = 0
    speeds.append(speed)
    
    I = integrate(rows_two[i],h) #Calculating the integral of the graph at each timestep
    integrals.append(I)
    
    print('time = %s, amplitude = %s, speed = %s, Integral = %s' %(time,amplitude,speed,I)) # printing amplitudes and speeds
    
    
    
#plotting amplitude of highest peak vs time and speed of highest peak vs time
plt.plot(times, amplitudes)
#plt.title('Maximum amplitude vs Time')
plt.xlabel('Time / s')
plt.ylabel('Amplitude / m')
plt.show()

plt.plot(times, speeds)
plt.title('Speed of the peak vs Time')
plt.xlabel('Time \ s')
plt.ylabel('Speed of the peak / m/s')
plt.show()

plt.plot(times, integrals) # you see a slight increase in the integral due to the oscillations behind x=0 increasing
#plt.title('Integral vs Time') # there is no sign of the collision changing the integral suggesting area under the graph remains unchanged
plt.xlabel('Time / s')
plt.ylabel('Integral')
plt.show()
#%%
#plotting the 2 similar speed soliton interaction
plt.plot(x_vals,rows_two[0], label='t=0s')
plt.plot(x_vals,rows_two[50], label='t=0.05s')
plt.plot(x_vals,rows_two[60],label='t=0.06s')
plt.plot(x_vals,rows_two[70],label='t=0.07s')
plt.plot(x_vals,rows_two[-1],label='t=0.1s')
plt.xlabel('x / m')
plt.ylabel('u / m')
plt.legend()
plt.xlim(-4,12)
plt.show()

#%%
#two solitons similar speeds
u_start = -10
u_end = 20

x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h))

ALPHA1 = 5
ALPHA2 = 4.5

numerator1 = (12 * ALPHA1 ** 2)
numerator2 = (12 * ALPHA2 ** 2)

u0_1 = numerator1 / (np.cosh(ALPHA1 * x_vals) ** 2)
u0_2 = numerator2 / (np.cosh(ALPHA2 * (x_vals - 1.5))**2) #starts the slow pulse at x=3

u0_two = u0_1 + u0_2
#plt.plot(x_vals, u0_two)

n_timesteps = 130

rows_two = [u0_two]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows_two[-1]
    fa_n = fa(un, h)
    fb_n = fb(un, fa_n, dt, h)
    fc_n = fc(un, fb_n, dt, h)
    fd_n = fd(un, fc_n, dt, h)
    
    rows_two.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)

speeds = []
amplitudes = []
times = []
integrals = []

for i in range(len(rows_two)): #plotting the function at successive time steps
    plt.plot(x_vals,rows_two[i])
    plt.ylim(0,300)
    plt.xlim(-1,20)
    plt.show()
    
    time = i * dt #calculating a plotting amplitude and speed vs time
    times.append(time)
    amplitude = max(rows_two[i])
    amplitudes.append(amplitude)
    if i >= 1:
        row = rows_two[i].tolist()
        row1 = rows_two[i-1].tolist()
        max_x_val_i = (row.index(max(rows_two[i])) * h) + u_start
        max_x_val_i1 = (row1.index(max(rows_two[i-1])) * h) + u_start
        speed = (max_x_val_i - max_x_val_i1) / dt
    else:
        speed = 0
    speeds.append(speed)
    
    I = integrate(rows_two[i],h) #Calculating the integral of the graph at each timestep
    integrals.append(I)
    
    print('time = %s, amplitude = %s, speed = %s, Integral = %s' %(time,amplitude,speed,I)) # printing amplitudes and speeds
    
    
    
#plotting amplitude of highest peak vs time and speed of highest peak vs time
plt.plot(times, amplitudes)
plt.title('Maximum amplitude vs Time')
plt.xlabel('Time / s')
plt.ylabel('Amplitude / m')
plt.show()

plt.plot(times, speeds)
plt.title('Speed of the peak vs Time')
plt.xlabel('Time \ s')
plt.ylabel('Speed of the peak / m/s')
plt.show()

plt.plot(times, integrals) # you see a slight increase in the integral due to the oscillations behind x=0 increasing
plt.title('Integral vs Time') # there is no sign of the collision changing the integral suggesting area under the graph remains unchanged
plt.xlabel('Time / s')
plt.ylabel('Integral')
plt.show()
#%%
#Wave breaking
#Using sech wave

def u0_sech(x_vals):
    ALPHA = 5
    numerator_sech = (12 * ALPHA ** 2)
    u0_sech = numerator_sech / (np.cosh(ALPHA * x_vals))
    return u0_sech

u0_sech = u0_sech(x_vals)

n_timesteps = 200
h = 0.1
dt = 0.0001
#now use this as the initial conditions
#un = u0 # starting approximation
rows_sech = [u0_sech]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows_sech[-1]
    fa_n = fa(un, h)
    fb_n = fb(un, fa_n, dt, h)
    fc_n = fc(un, fb_n, dt, h)
    fd_n = fd(un, fc_n, dt, h)
    
    rows_sech.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)

#for i in range(len(rows_sech)):
#    plt.plot(x_vals,rows_sech[i])
#    plt.show()
# using sech instead of sech squared as the input shows an output of two solitons separating
#%%
plt.plot(x_vals,rows_sech[0], label='t=0s')
plt.plot(x_vals,rows_sech[-1],label='t=0.02s')
plt.title('Wave breaking of u = 300sech(5x)')
plt.xlabel('x / m')
plt.ylabel('u / m')
plt.xlim(-5,8)
plt.legend()
plt.show()
#%%
#Wave breaking
#Using cos wave
#This block takes a long time to run
h = 0.1
x_vals_cos = np.linspace(-(np.pi), (np.pi), 400)
def u0_cos(x_vals):
    
    u0_cos = 1 - np.cos(x_vals)
    
    return u0_cos

u0_cos = u0_cos(x_vals_cos)

n_timesteps = 150000
dt = 0.0001
#now use this as the initial conditions
#un = u0 # starting approximation
rows_cos = [u0_cos]  # list of all calculated un approximations
for i in range(n_timesteps):
    un = rows_cos[-1]
    fa_n = fa(un, h)
    fb_n = fb(un, fa_n, dt, h)
    fc_n = fc(un, fb_n, dt, h)
    fd_n = fd(un, fc_n, dt, h)
    
    rows_cos.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)
    
#for i in range(len(rows_cos)):
#    plt.plot(x_vals,rows_cos[i])
#    plt.show()
#%%
plt.plot(x_vals_cos, rows_cos[0],label='t=0s')
plt.plot(x_vals_cos, rows_cos[50000],label='t=5s')
plt.plot(x_vals_cos, rows_cos[-1],label='t=15s')
plt.title('Wave breaking of u = 1 - cos(x)')
plt.ylabel('u')
plt.xlabel('x')
#plt.xlim(-np.pi, np.pi)
plt.legend()
plt.show()
#%%
#Shock Waves
n_timesteps = 3000
dt = 0.0001

alpha = 0.5 # alpha for runge-kutta
h = 0.1

ALPHA = 5 # alpha for u0 calculation

u_start = -10
u_end = 25

numerator = (12 * ALPHA ** 2)
x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h))
u0 = numerator / (np.cosh(ALPHA * x_vals) ** 2)

def f_sw(u,h):
    f_list = []
    for i in range(len(u)):
        # Calculate wrap around index for u_i
        i_p1 = (i + 1) % len(u)
#        i_p2 = (i + 2) % len(u)
        i_m1 = (i - 1) % len(u)
#        i_m2 = (i - 2) % len(u)
        
        t1 = -(1/(4*h)) * ((u[i_p1]**2) - (u[i_m1]**2))
#        t2 = -(1/(2*h**3)) * (u[i_p2] - 2*u[i_p1] + 2*u[i_m1] - u[i_m2])
                         
        f = t1
        f_list.append(f)
    return np.array(f_list)

def fa_sw(u, h):
    return f_sw(u, h)

# calculate fb given un
def fb_sw(u, fa, dt, h):
    return f_sw(u + 0.5 * dt * fa, h)

def fc_sw(u, fb, dt, h):
    return f_sw(u + 0.5 * dt * fb, h)

def fd_sw(u, fc, dt, h):
    return f_sw(u + dt * fc, h)

rows = [u0]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows[-1]
    fa_n = fa_sw(un, h)
    fb_n = fb_sw(un, fa_n, dt, h)
    fc_n = fc_sw(un, fb_n, dt, h)
    fd_n = fd_sw(un, fc_n, dt, h)
    
    rows.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)
    
#def sol_rk_4(n)
#diff_list = []
#for i in range(len(rows)):
#    plt.plot(x_vals,rows[i])
#    plt.plot(x_vals, sol(x_vals, i * dt, ALPHA))
#    plt.show()
    
plt.plot(x_vals, rows[0], label='t=0s')
plt.plot(x_vals, rows[1000], label='t=0.1s')
plt.plot(x_vals, rows[1900], label='t=0.19s')
#plt.title('Shock waves')
plt.xlabel('x / m')
plt.ylabel('u / m')
plt.legend()
plt.show()
# if you run it for longer instability is reached at 4716 time steps
# Need to introduce diffusive term
#%%
plt.plot(x_vals,rows[300])
#%%
#finding shockwave instability
n_timesteps = 50000
dt = 0.0001
rows = [u0]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows[-1]
    fa_n = fa_sw(un, h)
    fb_n = fb_sw(un, fa_n, dt, h)
    fc_n = fc_sw(un, fb_n, dt, h)
    fd_n = fd_sw(un, fc_n, dt, h)
    
    rows.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)
#unstable after 35392 time steps = 
#%%
#Introducing the diffusive term
n_timesteps = 10000
dt = 0.0001

alpha = 0.5 # alpha for runge-kutta
h = 0.1

ALPHA = 5 # alpha for u0 calculation

u_start = -10
u_end = 25

numerator = (12 * ALPHA ** 2)
x_vals = np.linspace (u_start, u_end, int((u_end - u_start) / h))
u0 = numerator / (np.cosh(ALPHA * x_vals) ** 2)

D = 0.01

def f_sw_D(u,h,D):
    f_list = []
    for i in range(len(u)):
        # Calculate wrap around index for u_i
        i_p1 = (i + 1) % len(u)
#        i_p2 = (i + 2) % len(u)
        i_m1 = (i - 1) % len(u)
#        i_m2 = (i - 2) % len(u)
        
        t1 = -(1/(4*h)) * ((u[i_p1]**2) - (u[i_m1]**2))
        t2 = (D/(h**2)) * (u[i_m1] - 2*u[i] + u[i_p1])
                         
        f = t1 + t2
        f_list.append(f)
    return np.array(f_list)

def fa_sw_D(u, h, D):
    return f_sw_D(u, h, D)

# calculate fb given un
def fb_sw_D(u, fa, dt, h, D):
    return f_sw_D(u + 0.5 * dt * fa, h, D)

def fc_sw_D(u, fb, dt, h, D):
    return f_sw_D(u + 0.5 * dt * fb, h, D)

def fd_sw_D(u, fc, dt, h, D):
    return f_sw_D(u + dt * fc, h, D)

rows = [u0]  # list of all calculated un approximations
for i in range(n_timesteps - 1):
    un = rows[-1]
    fa_n = fa_sw_D(un, h, D)
    fb_n = fb_sw_D(un, fa_n, dt, h, D)
    fc_n = fc_sw_D(un, fb_n, dt, h, D)
    fd_n = fd_sw_D(un, fc_n, dt, h, D)
    
    rows.append(un + (1/6) * (fa_n + 2 * fb_n + 2 * fc_n + fd_n) * dt)
#%%
plt.plot(x_vals,rows[0],label='t=0s')
plt.plot(x_vals,rows[500],label='t=0.05s')
plt.plot(x_vals,rows[1000],label='t=0.1s')
plt.plot(x_vals,rows[2000],label='t=0.2s')
#plt.plot(x_vals,rows[-1],label='t=1s')
plt.legend()
plt.xlim(-9,25)
plt.xlabel('x / m')
plt.ylabel('u / m')
plt.show()