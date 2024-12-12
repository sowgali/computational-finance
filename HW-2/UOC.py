import numpy as np
import math
import scipy.special as sc
import timeit
from tqdm import tqdm

S_0 = 1900
K = 2000
Barrier = 2200
r = 0.0025
q = 0.015
T = 0.5
base_vol = 0.25
nu = 0.31
theta = -0.25
Y = 0.4
SMIN = 500
Rebate = 0.0

xMin, xMax = np.log(SMIN), np.log(Barrier)
N = 800
M = 100
dx = (xMax - xMin) / N
dt = T / M
EPS = dx
tau = dt * np.arange(1, M+1)
x = xMin + np.arange(N+1) * dx

def g1(x, alpha):
    return sc.gammaincc(1-alpha, x) * sc.gamma(1-alpha)

def g2(x, alpha):
    return ((np.exp(-x) * (x**(-alpha)) / alpha)) - g1(x, alpha) / alpha

def sig_calculator(l):
    return (l**(Y-2)) * (-(l*EPS)**(1-Y) * np.exp(-l*EPS) + (1-Y) * (g1(0, Y) - g1(l*EPS, Y))) / nu

lambda_n = math.sqrt((theta**2/base_vol**4) + (2.0/(base_vol**2 * nu))) + theta/base_vol**2
lambda_p = math.sqrt((theta**2/base_vol**4) + (2.0/(base_vol**2 * nu))) - theta/base_vol**2

kx = np.arange(1, N+1) * dx
g1_n = g1(kx * lambda_n, Y)
g1_p = g1(kx * lambda_p, Y)
g2_n = g2(kx * lambda_n, Y)
g2_p = g2(kx * lambda_p, Y)
g2_n_plus = g2(kx * (lambda_n+1), Y)
g2_p_minus = g2(kx * (lambda_p-1), Y)

sigma = sig_calculator(lambda_n) + sig_calculator(lambda_p)
omega = ((lambda_p**Y) * g2(lambda_p*EPS, Y) - ((lambda_p-1)**Y * g2((lambda_p-1)*EPS, Y)) \
+ (lambda_n**Y) * g2(lambda_n*EPS, Y)  - ((lambda_n+1)**Y * g2((lambda_n+1)*EPS, Y))) / nu

alpha = sigma * dt / (2 * dx**2)
beta = r - q + omega - (sigma / 2)

Bl = alpha - beta * dt / (2*dx)
Bu = alpha + beta * dt / (2*dx)

print("sigma: ", sigma)
print("omega: ", omega)
print("Bl: ", Bl)
print("Bu: ", Bu)

def triDiag(LL, DD, UU, rhs):
    n = len(rhs)
    v = np.zeros(n)
    y = np.zeros(n)
    w = DD[0]
    y[0] = 1.0 * rhs[0] / w
    
    for i in range(1, n):
        v[i-1] = 1. * UU[i-1] / w
        w = DD[i] - LL[i] * v[i-1]
        y[i] = 1. * (rhs[i] - LL[i] * y[i-1]) / w
    
    for j in range(n-2, -1, -1):
        y[j] = y[j] - v[j] * y[j+1]
    
    return y


def sol(w):
    ans = np.zeros(N-1)
    for i in range(1, N):
        if i == 1 or i == N-1:
            ans[i-1] = 0
        else:
            for k in range(1, i):
                ans[i-1] += lambda_n**Y * (w[i-k] - w[i] - k * (w[i-k-1] - w[i-k])) * (g2_n[k-1] - g2_n[k])
                ans[i-1] += (w[i-k-1] - w[i-k]) * (g1_n[k-1] - g1_n[k]) / ((lambda_n ** (1-Y)) * dx)

            for k in range(1, N-i):
                ans[i-1] += lambda_p**Y * (w[i+k] - w[i] - k * (w[i+k+1] - w[i+k])) * (g2_p[k-1] - g2_p[k])
                ans[i-1] += (w[i+k-1] - w[i+k]) * (g1_p[k-1] - g1_p[k]) / ((lambda_p ** (1-Y)) * dx)
        ans[i-1] += K * lambda_n**Y * g2_n[i-1] - np.exp(x[i]) * (lambda_n + 1)**Y * g2_n_plus[i-1] 
    return ans


l = np.ones(N-1) * (-Bl)
u = np.ones(N-1) * (-Bu)
d = 1 + r*dt + Bu + Bl + dt * (lambda_n**Y * g2_n[:N-1] + lambda_p**Y * g2_p[::-1][:N-1]) / nu

u[-1] =  0
l[0] = 0

s = np.exp(x)
vCall = np.maximum(s - K, 0) * (s < Barrier)

start = timeit.default_timer()
for j in tqdm(range(M)):  
    rhs = (dt * sol(vCall) / nu) + vCall[1:N]
    inner = triDiag(l, d, u, rhs)
    vCall = np.pad(inner, (1, 1), 'constant', constant_values=(0, 0))
stop = timeit.default_timer()
print('Time: ', stop - start)

uoc_imp = np.interp(np.log(S_0), x, vCall)
print('Price of the UOC option:', uoc_imp)
