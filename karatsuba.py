# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:18:14 2019

@author: Mike
"""
x = 1234123412341234
y = 4321432143214321

def Karatsuba(x, y):
    if (x < 10) or (y < 10):
        return x*y
		
	#change to string for splitting
    xStr = str(x) 
    yStr = str(y)
	
    maxLen = max(len(xStr), len(yStr))
    splitPos = int(maxLen / 2)
	
    A, B= int(xStr[:-splitPos]), int(xStr[-splitPos:])
    C, D= int(yStr[:-splitPos]), int(yStr[-splitPos:])
	
	#recursion
    z0 = Karatsuba(B, D)
    z1 = Karatsuba((B + A), (D + C))
    z2 = Karatsuba(A, C)

    return (z2*10**(2*splitPos)) + ((z1-z2-z0)*10**(splitPos))+z0

k = Karatsuba(x,y)
print('Karatsuba: ',k)
print('Direct Mutiply: ',x*y)