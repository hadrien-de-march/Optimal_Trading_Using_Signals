#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 23:45:40 2018

@author: hadriendemarch
"""

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import fsolve

dt = 3.
q_0 = 1./dt
late = -1.

def f(x):
    addon = 1./dt*(1+1./(1-np.abs(x)))
    addon = max(1e-5, min(1-1e-5,addon))
    part_1 = x*(1+np.log((dt-1-late)*(1-np.abs(x))/(np.abs(x)))/dt)
    part_2 = 1-2*q_0-np.sign(1.-2*q_0)*(1+late*(1-np.abs(x)))/dt
    return part_1-part_2


def g(x):
    return (3+np.log(2+x))*(1-1./(3+x))-4.

sol  =fsolve(g, 0)
print(sol)

print(sol[0]-np.log(3))

def h(x):
    return (2+x)*(1+np.log(10+3*x)/3)-x*(10+3*x)/3

sol  =fsolve(h, 0)
print(sol)

print(sol[0]-np.log(3))


Xs = range(-100,100)
Xs = list(map(lambda x : x*1./100, Xs))
Ys = list(map(h, Xs))

plt.plot(Xs, Ys, label = "function")

plt.legend()
plt.show()