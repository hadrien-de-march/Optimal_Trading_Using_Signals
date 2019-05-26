# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:25:56 2014

@author: hdemarch
"""

import numpy as np

def f(K):
    return calc(K)
    

Nmax = 1000

Ymin = -1.

Ymax = 1.

Xmin = 0

Xmax = 3.14160


Value1 = np.zeros(Nmax)

Value2 = np.zeros(Nmax)

Value3 = np.zeros(Nmax)

Value4 = np.zeros(Nmax)

Value5 = np.zeros(Nmax)

for t in range(Nmax):
    
    Value1[t] =  max(Ymin,min(f(Xmin+(Xmax-Xmin)*t*1./Nmax)[0],Ymax))
    Value2[t] =  max(Ymin,min(f(Xmin+(Xmax-Xmin)*t*1./Nmax)[1],Ymax))
    Value3[t] =  max(Ymin,min(f(Xmin+(Xmax-Xmin)*t*1./Nmax)[2],Ymax))
    Value4[t] =  max(Ymin,min(f(Xmin+(Xmax-Xmin)*t*1./Nmax)[3],Ymax))
    Value5[t] =  max(Ymin,min(f(Xmin+(Xmax-Xmin)*t*1./Nmax)[4],Ymax))
    
plot(Value1)

#plot(Value2)
#
#plot(Value3)
#
#plot(Value4)
#
#plot(Value5)