"""
PHYS2124 Nano generator Lao Zhiquan
"""
import warnings
warnings.filterwarnings("ignore")
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmath
N=200
tau_eff=1/1.0
d_tilde_eff=5.0
"""
calculate a particular solution y_p
"""
def generate_matrix_p(f):#generate a matrix for equations to find Fourier coefficients
    ma=np.zeros((2*N+1,2*N+1),dtype=np.complex_)
    for n in range(-N,N+1):
        if n!=-N and n!=N:
            ma[n+N][n-1+N]=1/(4*tau_eff)
            ma[n+N][n+N]=-d_tilde_eff/tau_eff-1/(2*tau_eff)+(1j)*n*2*math.pi*f
            ma[n+N][n+1+N]=1/(4*tau_eff)
        if n==N:
            ma[n+N][n-1+N]=1/(4*tau_eff)
            ma[n+N][n+N]=-d_tilde_eff/tau_eff-1/(2*tau_eff)+(1j)*n*2*math.pi*f
        if n==-N:
            ma[n+N][n+N]=-d_tilde_eff/tau_eff-1/(2*tau_eff)+(1j)*n*2*math.pi*f
            ma[n+N][n+1+N]=1/(4*tau_eff)
    return ma
def generate_a_p(f):#calculate the Fourier coefficent
    sol_p=np.zeros((2*N+1,1),dtype=np.complex_)
    #a_0, a_1, and a_{-1} takes different values
    sol_p[0+N]=-1/(2*tau_eff)
    sol_p[N-1]=1/(4*tau_eff)
    sol_p[N+1]=1/(4*tau_eff)
    return np.linalg.lstsq(generate_matrix_p(f),sol_p)[0]
"""
calculate general solution of y
"""
def cal_y_gen(f,t):#calculate y_general
    term1=math.sin(2*math.pi*f*t)
    term2=2*math.pi*f*t*(2*d_tilde_eff+1)
    term3=4*math.pi*tau_eff*f
    #return d_tilde_eff/tau_eff*math.exp((term1-term2)/term3)
    return math.exp((term1-term2)/term3)
"""
calculate y(f,t)
"""
def calculate_constant(a_p):#y(t)=const*y_general+y_p, calculate the constant
    sum_a_p=0
    for i in range(len(a_p)):
        sum_a_p+=a_p[i][0]
    #return -sum_a_p/(d_tilde_eff/tau_eff)
    return -sum_a_p
def calculate_y(a_p,const,f,t):#calculate y(t)
    ans=0
    for n in range(-N,N+1):
        ans+=cmath.exp((-1j)*n*2*math.pi*f*t)*a_p[n+N][0]
    ans=ans+cal_y_gen(f,t)*const
    ans=round(ans,15)#eliminate small imaginary part due to float storage
    return ans
"""
Q1
"""
a_p=generate_a_p(10)#generate Fourier coefficents at f=10Hz
const=calculate_constant(a_p)
y=[]
time=[]
for t in np.arange(0,2.05,0.01):
    y.append(calculate_y(a_p, const, 10, t))#calculate y
    time.append(t)
plt.xlabel('t')
plt.ylabel('y')
plt.title('y(t) at f=10Hz')
plt.plot(time,y)
"""
Q2
The plot is taken when sigma_e=1, d_1=2, d_2=5, epsilon_1=3e-11, epsilon_2=7e-11, epsilon_0=8.85e-12
"""

sigma_e=1
d_1=2
d_2=5
epsilon_1=3e-11
epsilon_2=7e-11
epsilon_0=8.85e-12
d_max=epsilon_0/d_tilde_eff*(d_1/epsilon_1+d_2/epsilon_2)
R=1
A=tau_eff*d_max/(epsilon_0*R)
"""
#constants tried
d_max=5
sigma_e=1
R=1
epsilon_0=8.85e-12
"""

def calculate_U(f,t):
    a_p=generate_a_p(f)
    const=calculate_constant(a_p)
    y=calculate_y(a_p, const, f, t)
    d_t=d_max*0.5*(1-math.cos(2*math.pi*f*t))
    return d_1*y*sigma_e/epsilon_1+d_2*y*sigma_e/epsilon_2-d_t*(sigma_e-y*sigma_e)/epsilon_0
    #return d_max*y-d_t*(sigma_e-y*sigma_e)/epsilon_0
w=[]
freq=[]
for f in np.arange(0,10,0.01):
    w.append(calculate_U(f, 5)*calculate_U(f, 5)/R)
    freq.append(f)
plt.clf()
plt.xlabel('f')
plt.ylabel('W')
plt.title('W(f) at t=5s')
plt.plot(freq,w)

"""
Q3
for large t, y_gen can be negelected
"""
def cal_y_p(a_p,f,t):
    ans=0
    for n in range(-N,N+1):
        ans+=cmath.exp((-1j)*n*2*math.pi*f*t)*a_p[n+N][0]
    ans=round(ans,15)#eliminate small imaginary part due to float storage
    return ans
def cal_U_large(a_p,f,t):
    d_t=d_max*0.5*(1-math.cos(2*math.pi*f*t))
    y=cal_y_p(a_p,f,t)
    return d_1*y*sigma_e/epsilon_1+d_2*y*sigma_e/epsilon_2-d_t*(sigma_e-y*sigma_e)/epsilon_0
#calculate the integration
def cal_w_avg(T,a_p):
    W_avg=0
    for t in np.arange(0,T,0.01):
        W_avg+=cal_U_large(a_p,10,t)*cal_U_large(a_p,10,t)/R*0.01
    W_avg=W_avg/T
    return W_avg
Time=[]
w_avg=[]
for T in np.arange(10,20,0.1):
    Time.append(T)
    w_avg.append(cal_w_avg(T,a_p))
plt.clf()
plt.xlabel('T')
plt.ylabel('w_avg')
plt.plot(Time,w_avg)
plt.title('w_avg against T')