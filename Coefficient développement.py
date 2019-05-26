# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:19:49 2014

@author: hdemarch
"""

import sympy as sp

import numpy as np

from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction

def nCk(n,k): 
  return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )
  
def fact(n):
    return Fact_grid[n]

Nfact = 10
    
Fact_grid = np.zeros(Nfact+1)

Fact_grid+=1.


for k in range(Nfact):
    Fact_grid[k+1] = Fact_grid[k]*(k+1)

sgm = sp.symbols('sigma')

sgm = 1.

N = 200

Correct = 0

coeff = sp.zeros((N,1))
coeffApp = sp.zeros((N,1))

Coeff = np.zeros((N,1))
CoeffApp = np.zeros((N,1))

for k in range(N/2):
    print(k)
    k += 1
    coeff[2*k-2] = 0
    somme = 0
    for i in range(k-1):
        q=i+2
        somme += 2**(q)*nCk(k,q)*coeff[2*k-q]
    coeff[2*k-1] = ((-1)**(k)-somme)*1./(2*k)
    coeffApp[2*k-1] = (-1)**(k)*1./(2.*k)



for k in range(Correct+1):
    coeffApp[k]=coeff[k]
    
for k in range(N):
    Coeff[k] = coeff[k]
    CoeffApp[k] = coeffApp[k]
    for l in range(k):
        Coeff[k] = Coeff[k]*1./(k-l)
        CoeffApp[k] = CoeffApp[k]*1./(k-l)

    
def fap(x):
    if x==0:
        return 0
    else:
        return -(1-np.cos(x/sgm**2))/(x/sgm**2)
    
def fap_plus(x):
    if x==0:
        return 0
    else:
        return -(1-np.cos(x/sgm**2))/(x/sgm**2)-1./12*x**5/(fact(5)*sgm**6)
    
def fap_plus_plus(x):
    if x==0:
        return 0
    else:
        return -(1-np.cos(x/sgm**2))/(x/sgm**2)-1./12*x**5/(fact(5)*sgm**6)+1./(12*sgm**8)*x**7/fact(7) #+1./(24*sgm**4)*x**7/fact(7)
    
def fap_plus_plus_plus(x):
    if x==0:
        return 0
    else:
        return -(1-np.cos(x/sgm**2))/(x/sgm**2)-2.**(5./2)/(12*sgm**11)*(np.sin(x*sgm/np.sqrt(2))-x*sgm/np.sqrt(2)+(x*sgm/np.sqrt(2))**3/6)+1./(12*sgm**8)*x**7/fact(7)

def calc(x):
    X = np.zeros((N,1))
    for k in range(N):
        X[k]=(x/sgm**2)**k
    return [np.sum(X*Coeff),np.sum(X*CoeffApp),fap(x),fap_plus(x),fap_plus_plus(x)]
    

print calc(2.)
a = np.zeros(N)
