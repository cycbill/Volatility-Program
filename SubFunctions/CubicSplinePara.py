import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CubicSplineVol(X, XQuotes, a, b, c, d):
    low = 0
    high = len(
        XQuotes) - 1
    while high - low > 1:
        j = int((low + high) / 2)
        if X >= min(XQuotes[j], XQuotes[high] ) and X <= max(XQuotes[j], XQuotes[high] ):
            low = j
        elif X >= min(XQuotes[j], XQuotes[low] ) and X <= max(XQuotes[j], XQuotes[low] ):
            high = j
    if low < high:
        k = low
    else:
        k = high
    Y = a[k] * (X - XQuotes[k])**3 + \
        b[k] * (X - XQuotes[k])**2 + \
        c[k] * (X - XQuotes[k]) + d[k]
    return Y


def LinearExtrapolateVol(X, X0, X1, Y0, Y1):
    Y = (Y0 - Y1) / (X0 - X1) * (X - X0) + Y0
    return Y


def CubicSplinePara(X, Y, Mode):
    ## Data initialization
    NX = len(X)
    n = NX - 1
    h = np.diff(X)
    hMat = np.zeros((NX, NX))
    S = np.zeros(NX)

    ## Calculation of (NX-1)*(NX-1) Coefficient Matrix and the Column Matrix
    if Mode == 'Natural':
        hMat[0,0] = 1
    elif Mode == 'Clamped':
        hMat[0,0] = 2 * h[0]
        hMat[0,1] = h[0]

    for i in range(1, NX-1):
        hMat[i,i-1] = h[i-1]
        hMat[i,i]   = ( h[i-1] + h[i] ) * 2
        hMat[i,i+1] = h[i]

    if Mode == 'Natural':    
        hMat[-1,-1] = 1
    elif Mode == 'Clamped':
        hMat[-1,-1] = 2 * h[-1]
        hMat[-1,-2] = h[-1]

    divdif = np.diff(Y) / h
    dMat = np.zeros(NX)
    dMat[1:n] = ( 3 * np.diff(divdif) ) 
    if Mode == 'Clamped':
        dMat[0] = 3 * divdif[0] - 3 * (Y[0]-Y[1])/(X[0]-X[1])
        dMat[-1] = 3 * (Y[-1]-Y[-2])/(X[-1]-X[-2]) - 3 * divdif[-1]

    S = np.linalg.solve(hMat, dMat)
    
    ## Calculation of the (N-1) Cubic curves' Coefficients
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)      
    for i in range(0, n):
        a[i] = ( S[i+1] - S[i] ) / ( 3 * h[i] )
        b[i] = S[i]
        c[i] = ( Y[i+1] - Y[i] ) / h[i] - h[i]*( 2*S[i] + S[i+1] ) / 3
        d[i] = Y[i]

    #y[x] = a[j](x-X[j])^3 + b[j](x-X[j])^2 + c[j](x-X[j]) + d[j]
    return a, b, c, d 
