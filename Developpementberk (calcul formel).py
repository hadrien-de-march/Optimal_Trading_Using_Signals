# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2014

@author: hdemarch
"""

import sympy as sp

X = sp.symbols('X')

al = sp.symbols('Alp')

d = sp.symbols('delt')

t = sp.symbols('Tau')

a=sp.symbols('a')

b=sp.symbols('b')

c = sp.symbols('c')

f = a/t**2 + b/t + c

df = sp.symbols('df')

eta = 2*d*al**2/t

Exp = -(-al**2*X**2/t**2+X*df)*(al-eta)**2+1./4*((al-eta)**2+(al**2*(2*X/t-1.)-f))**2

Dev = sp.expand(Exp)

print(Dev)


#e = K/8.
#
#Delta = 256.*e**3-128.*c**2*e**2+144.*c*d**2*e-27.*d**4+16.*c**4*e-4.*c**3*d**2
#
#Delta0 = c**2+12.*e
#
#Delta1 = 2.*c**3+27.*d**2-72.*c*e
#
#Q = ((Delta1+(-27.*Delta)**(1./2))/2.)**(1./3)
#
#S = 1./2*(-2./3*c+1./3*(Q+Delta0/Q))**(1./2)
#
#Sol = S+1./2*(-4*S**2-2*c-d/S)**(1./2)
#
#Exp = sp.expand(Sol)
#
##print(Exp)
#
#Alp = sp.symbols()