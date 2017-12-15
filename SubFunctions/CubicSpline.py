import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt 
# import from my own packages
from SubFunctions.CubicSplinePara import CubicSplineVol, LinearExtrapolateVol, CubicSplinePara
from SubFunctions.OptionPricing import OptionDelta

def CubSpline_LinExt_DeltaIntrp(DIntrp, X, Y):

    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(X, Y, 'Clamped')

    ## Initialization
    NX = len(X)
    NIntrp = len(DIntrp)
    YIntrp = np.zeros_like(DIntrp)

    ## Calculate Interpolated Vols
    for i in range(NIntrp):
        if DIntrp[i] >= X[0]:
            YIntrp[i] = LinearExtrapolateVol(DIntrp[i], X[0], X[1], Y[0], Y[1])
        elif DIntrp[i] <= X[-1]:
            YIntrp[i] = LinearExtrapolateVol(DIntrp[i], X[-1], X[-2], Y[-1], Y[-2])
        else:
            YIntrp[i] = CubicSplineVol(DIntrp[i], X, a,b,c,d)

    return YIntrp, a, b, c, d


def CubSpline_DeltaIntrp_Solving(K, X, Y, CPSign, S, VAtm, Rd, Rf, Te, Td, Premium, DeltaMode ):

    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(X, Y, 'Clamped')

    Delta = lambda V: OptionDelta(CPSign, S, K, V, Rd, Rf, Te, Td, Premium, DeltaMode )
    VIntrp = lambda V: CubicSplineVol(Delta(V), X, a,b,c,d)
    V_Diff = lambda V: VIntrp(V) - V

    V_K = opt.newton(V_Diff, VAtm, tol = 1e-4)
    return V_K, a,b,c,d



def CubSpline_LogMon9P(X_XIntrp, XQuotes, VQuotes ):

    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(XQuotes, VQuotes, 'Clamped')

    ## Initialization
    NX = len(XQuotes)
    NIntrp = X_XIntrp.size
    V_XIntrp = np.zeros_like(X_XIntrp)

    ## Calculate Interpolated Vols
    for i in range(NIntrp):
        if X_XIntrp[i] >= XQuotes[0]:
            V_XIntrp[i] = LinearExtrapolateVol(X_XIntrp[i], XQuotes[0], XQuotes[1], VQuotes[0], VQuotes[1])
        elif X_XIntrp[i] <= XQuotes[-1]:
            V_XIntrp[i] = LinearExtrapolateVol(X_XIntrp[i], XQuotes[-1], XQuotes[-2], VQuotes[-1], VQuotes[-2])
        else:
            V_XIntrp[i] = CubicSplineVol(X_XIntrp[i], XQuotes, a,b,c,d)


    return V_XIntrp, a, b, c, d    


def CubSpline_LogMon5P_Flat(X_XIntrp, XQuotes, VQuotes ):
    
    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(XQuotes, VQuotes, 'Natural')

    ## Initialization
    NX = len(XQuotes)
    NIntrp = len(X_XIntrp)
    V_XIntrp = np.zeros_like(X_XIntrp)

    ## Calculate Interpolated Vols
    for i in range(NIntrp):
        if X_XIntrp[i] >= XQuotes[0]:
            V_XIntrp[i] = VQuotes[0]
        elif X_XIntrp[i] <= XQuotes[-1]:
            V_XIntrp[i] = VQuotes[-1]
        else:
            V_XIntrp[i] = CubicSplineVol(X_XIntrp[i], XQuotes, a,b,c,d)


    return V_XIntrp, a, b, c, d    

def CubSpline_LogMon5P_Linear(X_XIntrp, XQuotes, VQuotes ):
        
    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(XQuotes, VQuotes, 'Clamped')

    ## Initialization
    NX = len(XQuotes)
    NIntrp = len(X_XIntrp)
    V_XIntrp = np.zeros_like(X_XIntrp)

    ## Calculate Interpolated Vols
    for i in range(NIntrp):
        if X_XIntrp[i] >= XQuotes[0]:
            V_XIntrp[i] = LinearExtrapolateVol(X_XIntrp[i], XQuotes[0], XQuotes[1], VQuotes[0], VQuotes[1])
        elif X_XIntrp[i] <= XQuotes[-1]:
            V_XIntrp[i] = LinearExtrapolateVol(X_XIntrp[i], XQuotes[-1], XQuotes[-2], VQuotes[-1], VQuotes[-2])
        else:
            V_XIntrp[i] = CubicSplineVol(X_XIntrp[i], XQuotes, a,b,c,d)


    return V_XIntrp, a, b, c, d   

def CubSpline_Strike_Linear(KIntrp, KQuotes, VQuotes ):
        
    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(KQuotes, VQuotes, 'Clamped')

    ## Initialization
    NX = len(KQuotes)
    NIntrp = len(KIntrp)
    V_XIntrp = np.zeros_like(KIntrp)

    ## Calculate Interpolated Vols
    for i in range(NIntrp):
        if KIntrp[i] <= KQuotes[0]:
            V_XIntrp[i] = LinearExtrapolateVol(KIntrp[i], KQuotes[0], KQuotes[1], VQuotes[0], VQuotes[1])
        elif KIntrp[i] >= KQuotes[-1]:
            V_XIntrp[i] = LinearExtrapolateVol(KIntrp[i], KQuotes[-1], KQuotes[-2], VQuotes[-1], VQuotes[-2])
        else:
            V_XIntrp[i] = CubicSplineVol(KIntrp[i], KQuotes, a,b,c,d)


    return V_XIntrp, a, b, c, d   
