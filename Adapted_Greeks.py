from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt 
from scipy.interpolate import splrep, splev
from openpyxl import Workbook
# import from my own packages
from SubFunctions.OptionPricing import StrikeSolver, ATMStrike, OptionPrice, OptionDelta
from SubFunctions.BrokerToSmile import BrokerToSmile
from SubFunctions.CubicSplinePara import CubicSplinePara, LinearExtrapolateVol, LinearExtrapolateVol, CubicSplineVol
from SubFunctions.ReadData import ReadData

def CubSpline_DeltaIntrp_Solving(K, KQuotes, DeltaQuotes, VQuotes, CPSign, S, VAtm, Rd, Rf, Te, Td, Premium, DeltaMode ):

    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(DeltaQuotes, VQuotes, 'Clamped')

    Delta = lambda V: OptionDelta(CPSign, S, K, V, Rd, Rf, Te, Td, Premium, DeltaMode )

    if K < KQuotes[0]:
        VIntrp = lambda V: LinearExtrapolateVol(Delta(V), DeltaQuotes[0], DeltaQuotes[1],VQuotes[0], VQuotes[1])
        V_Start = VQuotes[0]
    elif K > KQuotes[-1]:
        VIntrp = lambda V: LinearExtrapolateVol(Delta(V), DeltaQuotes[-1], DeltaQuotes[-2],VQuotes[-1], VQuotes[-2])
        V_Start = VQuotes[-1]
    else:
        low = 0
        high = len(KQuotes) - 1
        while high - low > 1:
            j = int((low + high) / 2)
            if K >= min(KQuotes[j], KQuotes[high] ) and K <= max(KQuotes[j], KQuotes[high] ):
                low = j
            elif K >= min(KQuotes[j], KQuotes[low] ) and K <= max(KQuotes[j], KQuotes[low] ):
                high = j
        if low < high:
            m = low
        else:
            m = high
        VIntrp = lambda V: a[m] * (Delta(V) - DeltaQuotes[m])**3 + \
                            b[m] * (Delta(V) - DeltaQuotes[m])**2 + \
                            c[m] * (Delta(V) - DeltaQuotes[m]) + d[m]
        V_Start = VQuotes[m]

    V_Diff = lambda V: VIntrp(V) - V

    V_K = opt.newton(V_Diff, V_Start, tol = 1e-4)
    return V_K

def CubSpline_LogMon9P(X_XIntrp, XQuotes, VQuotes ):

    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(XQuotes, VQuotes, 'Clamped')

    ## Calculate Interpolated Vols
    if X_XIntrp >= XQuotes[0]:
        V_XIntrp = LinearExtrapolateVol(X_XIntrp, XQuotes[0], XQuotes[1], VQuotes[0], VQuotes[1])
    elif X_XIntrp <= XQuotes[-1]:
        V_XIntrp = LinearExtrapolateVol(X_XIntrp, XQuotes[-1], XQuotes[-2], VQuotes[-1], VQuotes[-2])
    else:
        V_XIntrp = CubicSplineVol(X_XIntrp, XQuotes, a,b,c,d)

    return V_XIntrp

def CubSpline_LogMon5P_Flat(X_XIntrp, XQuotes, VQuotes ):
    
    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(XQuotes, VQuotes, 'Natural')

    ## Calculate Interpolated Vols
    if X_XIntrp >= XQuotes[0]:
        V_XIntrp = VQuotes[0]
    elif X_XIntrp <= XQuotes[-1]:
        V_XIntrp = VQuotes[-1]
    else:
        V_XIntrp = CubicSplineVol(X_XIntrp, XQuotes, a,b,c,d)

    return V_XIntrp

def CubSpline_LogMon5P_Linear(X_XIntrp, XQuotes, VQuotes ):
        
    ## Calculate Cubic Spline Function Parameters
    a,b,c,d = CubicSplinePara(XQuotes, VQuotes, 'Clamped')


    ## Calculate Interpolated Vols
    if X_XIntrp >= XQuotes[0]:
        V_XIntrp = LinearExtrapolateVol(X_XIntrp, XQuotes[0], XQuotes[1], VQuotes[0], VQuotes[1])
    elif X_XIntrp <= XQuotes[-1]:
        V_XIntrp = LinearExtrapolateVol(X_XIntrp, XQuotes[-1], XQuotes[-2], VQuotes[-1], VQuotes[-2])
    else:
        V_XIntrp = CubicSplineVol(X_XIntrp, XQuotes, a,b,c,d)

    return V_XIntrp

def Adapted_Greeks(S, K, MatLabel, Te, Td, Rd, Rf, VAtm, RR25, BF25, RR10, BF10, ATMMode, Premium, DeltaMode, StrangleMode):
    ## 2. Market Data Pre-Processing
    F = S * np.exp( (Rd - Rf) * Td )
    DFf = np.exp(-Rf * Te)
    DFd = np.exp(-Rd * Te)
    plt.style.use('ggplot')

    ## 3. ATM Delta & Strike
    DeltaAtm = OptionDelta(-1, S, F, VAtm, Rd, Rf, Te, Td, Premium, DeltaMode)
    #print('Premium included fwd delta = {}'.format(DeltaAtm))
    KAtm     = ATMStrike(S, VAtm, Rd, Rf, Te, Td, Premium, ATMMode )
    #print('ATM Strike = {}'.format(KAtm))
    #print('\n')

    ## 4. 1-vol quote to 2-vol quote convertion
    if StrangleMode == '1vol':
        [K25c1v, K25p1v, V25c1v, V25p1v, K25c2v, K25p2v, V25c2v, V25p2v] = BrokerToSmile(0.25, VAtm, RR25, BF25, S, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
        print([K25c1v, K25p1v, V25c1v, V25p1v, K25c2v, K25p2v, V25c2v, V25p2v])
        #print('K25c1v = {0}, V25c1v = {1}.'.format(K25c1v, V25c1v))
        #print('K25p1v = {0}, V25p1v = {1}.'.format(K25p1v, V25p1v))
        #print('K25c2v = {0}, V25c2v = {1}.'.format(K25c2v, V25c2v))
        #print('K25p2v = {0}, V25p2v = {1}.'.format(K25p2v, V25p2v))
        #print('\n')

        [K10c1v, K10p1v, V10c1v, V10p1v, K10c2v, K10p2v, V10c2v, V10p2v] = BrokerToSmile(0.10, VAtm, RR10, BF10, S, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
        #print('K10c1v = {0}, V10c1v = {1}.'.format(K10c1v, V10c1v))
        #print('K10p1v = {0}, V10p1v = {1}.'.format(K10p1v, V10p1v))
        #print('K10c2v = {0}, V10c2v = {1}.'.format(K10c2v, V10c2v))
        #print('K10p2v = {0}, V10p2v = {1}.'.format(K10p2v, V10p2v))
        #print('\n')
    elif StrangleMode == '2vol':
        V25c2v = VAtm + BF25 + RR25 / 2
        V25p2v = VAtm + BF25 - RR25 / 2
        V10c2v = VAtm + BF10 + RR10 / 2
        V10p2v = VAtm + BF10 - RR10 / 2
        K25c2v = StrikeSolver( 0.25,  1, S, V25c2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
        K25p2v = StrikeSolver(-0.25, -1, S, V25p2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
        K10c2v = StrikeSolver( 0.10,  1, S, V10c2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
        K10p2v = StrikeSolver(-0.10, -1, S, V10p2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
        #print('K25c2v = {0}, V25c2v = {1}.'.format(K25c2v, V25c2v))
        #print('K25p2v = {0}, V25p2v = {1}.'.format(K25p2v, V25p2v))
        #print('K10c2v = {0}, V10c2v = {1}.'.format(K10c2v, V10c2v))
        #print('K10p2v = {0}, V10p2v = {1}.'.format(K10p2v, V10p2v))

    ## 5. Extra Quotes for Log Moneyness-9 points
    WingRatioP = 2.3

    WingRatioC = 2.3
    V1p = V25p2v + WingRatioP * abs( V10p2v - V25p2v )   # 1 delta put vol
    V001p = V1p                                         # 0.01 delta put vol
    V1c = V25c2v + WingRatioC * abs( V10c2v - V25c2v )   # 1 delta call vol
    V001c = V1c                                         # 0.01 delta call vol

    K001p = StrikeSolver(-0.01/100, -1, S, V001p, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    K1p   = StrikeSolver(-1/100,    -1, S, V1p,   Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    K001c = StrikeSolver(0.01/100,   1, S, V001c, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    K1c   = StrikeSolver(1/100,      1, S, V1c,   Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)

    

    ########################################### Interpolation ################################################

    ## 6. Interpolated Delta Points
    ## 6-a. Delta Intrp
    
    ND = 5
    VQuotes5 = np.array([V10p2v, V25p2v, VAtm, V25c2v, V10c2v])
    DeltaQuotes5 = np.array([-10, -25, DeltaAtm*100, 25, 10]) / 100
    KQuotes5 = np.array([K10p2v, K25p2v, KAtm, K25c2v, K10c2v])
    PutDeltaQuotes5 = OptionDelta(-1, S, KQuotes5, VQuotes5, Rd, Rf, Te, Td, 'Excluded', DeltaMode )
    K_DIntrp = K
    
    V_DIntrp = CubSpline_DeltaIntrp_Solving(K_DIntrp, KQuotes5, PutDeltaQuotes5, VQuotes5, -1, S, VAtm, Rd, Rf, Te, Td, 'Excluded', DeltaMode )
    PremC_DIntrp = OptionPrice( 1, S, K_DIntrp, V_DIntrp,  Rd, Rf, Te, Td)   # Call Premium
    PremP_DIntrp = OptionPrice(-1, S, K_DIntrp, V_DIntrp, Rd, Rf, Te, Td)   # Put Premium
    
    ## 6-b. Log Moneyness-9 Point Intrp
    NX = 9
    VQuotes9 = np.array([V001p, V1p, V10p2v, V25p2v, VAtm, V25c2v, V10c2v, V1c, V001c])
    DeltaQuotes9 = np.array([-0.01, -1, -10, -25, DeltaAtm*100, 25, 10, 1, 0.01]) / 100
    KQuotes9 = np.array([K001p, K1p, K10p2v, K25p2v, KAtm, K25c2v, K10c2v, K1c, K001c])
    PutDeltaQuotes9 = OptionDelta(-1, S, KQuotes9, VQuotes9, Rd, Rf, Te, Td, 'Excluded', DeltaMode )
    XQuotes9 = np.log(F / KQuotes9) / (VAtm**2 * Te)
    K_XIntrp = K
    X_XIntrp9 = np.log(F / K_XIntrp) / (VAtm**2 * Te)
    V_XIntrp9 = CubSpline_LogMon9P(X_XIntrp9, XQuotes9, VQuotes9 )
    D_XIntrp9 = OptionDelta(-1, S, K_XIntrp, V_XIntrp9, Rd, Rf, Te, Td, 'Excluded', DeltaMode )         # Delta
    PremC_XIntrp9 = OptionPrice( 1, S, K_XIntrp, V_XIntrp9,  Rd, Rf, Te, Td)   # Call Premium
    PremP_XIntrp9 = OptionPrice(-1, S, K_XIntrp, V_XIntrp9, Rd, Rf, Te, Td)   # Put Premium
    
    ## 6-c. Log Moneyness-5 Point Flat
    XQuotes5 = np.log(F / KQuotes5) / (VAtm**2 * Te)
    #K_XIntrp = np.linspace(K001p, K001c, 200, endpoint=True)
    X_XIntrp5F = np.log(F / K_XIntrp) / (VAtm**2 * Te)
    V_XIntrp5F = CubSpline_LogMon5P_Flat(X_XIntrp5F, XQuotes5, VQuotes5 )
    D_XIntrp5F = OptionDelta(-1, S, K_XIntrp, V_XIntrp5F, Rd, Rf, Te, Td, 'Excluded', DeltaMode )          # Delta
    PremC_XIntrp5F = OptionPrice( 1, S, K_XIntrp, V_XIntrp5F,  Rd, Rf, Te, Td)   # Call Premium
    PremP_XIntrp5F = OptionPrice(-1, S, K_XIntrp, V_XIntrp5F, Rd, Rf, Te, Td)   # Put Premium

    ## 6-d. Log Moneyness-5 Point Linear
    #K_XIntrp = np.linspace(K001p, K001c, 200, endpoint=True)
    X_XIntrp5L = np.log(F / K_XIntrp) / (VAtm**2 * Te)
    V_XIntrp5L = CubSpline_LogMon5P_Linear(X_XIntrp5L, XQuotes5, VQuotes5 )
    D_XIntrp5L = OptionDelta(-1, S, K_XIntrp, V_XIntrp5L, Rd, Rf, Te, Td, 'Excluded', DeltaMode )          # Delta
    PremC_XIntrp5L = OptionPrice( 1, S, K_XIntrp, V_XIntrp5L,  Rd, Rf, Te, Td)   # Call Premium
    PremP_XIntrp5L = OptionPrice(-1, S, K_XIntrp, V_XIntrp5L, Rd, Rf, Te, Td)   # Put Premium
    

    return PremP_DIntrp, PremP_XIntrp9, PremP_XIntrp5F, PremP_XIntrp5L



## 1. Read Data
[ccypair, S, MatLabel, Te, Td, Rd, Rf, 
 VAtm, RR25, BF25, RR10, BF10, 
 ATMMode, Premium, DeltaMode, StrangleMode] = ReadData()

if ccypair == 'EURUSD':
    ## EUR/USD
    K = 1.043000
    Sx = np.linspace(0.930616826496, 1.21134157859, 100, endpoint=True)
elif ccypair == 'USDJPY':
    ## USD/JPY
    K = 85.9718
    Sx = np.linspace(48.0847143676, 187.694032047, 100, endpoint=True)


Sx1 = Sx * (1 + 0.0001)
Sx2 = Sx * (1 - 0.0001)

len_Sx = len(Sx)


PremP_DIntrp = np.zeros(len_Sx); PremP_XIntrpW = np.zeros(len_Sx); PremP_XIntrpF = np.zeros(len_Sx); PremP_XIntrpL = np.zeros(len_Sx)
PremP_DIntrp1 = np.zeros(len_Sx); PremP_XIntrpW1 = np.zeros(len_Sx); PremP_XIntrpF1 = np.zeros(len_Sx); PremP_XIntrpL1 = np.zeros(len_Sx)
PremP_DIntrp2 = np.zeros(len_Sx); PremP_XIntrpW2 = np.zeros(len_Sx); PremP_XIntrpF2 = np.zeros(len_Sx); PremP_XIntrpL2 = np.zeros(len_Sx)


for i in range(len_Sx):
    PremP_DIntrp[i], PremP_XIntrpW[i], PremP_XIntrpF[i], PremP_XIntrpL[i]    = Adapted_Greeks(Sx[i],  K, MatLabel, Te, Td, Rd, Rf, VAtm, RR25, BF25, RR10, BF10, ATMMode, Premium, DeltaMode, StrangleMode)
    PremP_DIntrp1[i], PremP_XIntrpW1[i], PremP_XIntrpF1[i], PremP_XIntrpL1[i] = Adapted_Greeks(Sx1[i], K, MatLabel, Te, Td, Rd, Rf, VAtm, RR25, BF25, RR10, BF10, ATMMode, Premium, DeltaMode, StrangleMode)
    PremP_DIntrp2[i], PremP_XIntrpW2[i], PremP_XIntrpF2[i], PremP_XIntrpL2[i] = Adapted_Greeks(Sx2[i], K, MatLabel, Te, Td, Rd, Rf, VAtm, RR25, BF25, RR10, BF10, ATMMode, Premium, DeltaMode, StrangleMode)

Sx_Diff = Sx1 - Sx2
Delta_DIntrp = ( PremP_DIntrp1 - PremP_DIntrp2 ) / Sx_Diff
Delta_XIntrpW = ( PremP_XIntrpW1 - PremP_XIntrpW2 ) / Sx_Diff
Delta_XIntrpF = ( PremP_XIntrpF1 - PremP_XIntrpF2 ) / Sx_Diff
Delta_XIntrpL = ( PremP_XIntrpL1 - PremP_XIntrpL2 ) / Sx_Diff
Gamma_DIntrp = ( PremP_DIntrp1 - 2 * PremP_DIntrp + PremP_DIntrp2 ) / Sx_Diff / Sx_Diff
Gamma_XIntrpW = ( PremP_XIntrpW1 - 2 * PremP_XIntrpW + PremP_XIntrpW2 ) / Sx_Diff / Sx_Diff
Gamma_XIntrpF = ( PremP_XIntrpF1 - 2 * PremP_XIntrpF + PremP_XIntrpF2 ) / Sx_Diff / Sx_Diff
Gamma_XIntrpL = ( PremP_XIntrpL1 - 2 * PremP_XIntrpL + PremP_XIntrpL2 ) / Sx_Diff / Sx_Diff


###################################################### Plotting for Essay ############################################################

## Delta Interpolation scale with Linear Extrapolation
plt.figure(figsize=(14,6))
plt.suptitle('Delta Interpolation Scale with Linear Extrapolation',fontsize=16, fontweight='bold')
## Adapted Delta vs Spot
plt.subplot(121)
plt.plot(Sx, Delta_DIntrp,'.-')
plt.title('Adapted Delta vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Delta',color='black')
## Adapted Gamma vs Spot
plt.subplot(122)
plt.plot(Sx, Gamma_DIntrp,'.-')
plt.title('Adapted Gamma vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Gamma',color='black')
plt.show()


## Adapted Log Invert Moneyness Interpolation Scale with Flat Extrapolation
plt.figure(figsize=(14,6))
plt.suptitle('Adapted Log Invert Moneyness Interpolation Scale with Flat Extrapolation',fontsize=16, fontweight='bold')
## Adapted Delta vs Spot
plt.subplot(121)
plt.plot(Sx, Delta_XIntrpF,'.-')
plt.title('Adapted Delta vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Delta',color='black')
## Adapted Gamma vs Spot
plt.subplot(122)
plt.plot(Sx, Gamma_XIntrpF,'.-')
plt.title('Adapted Gamma vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Gamma',color='black')
plt.show()


## Adapted Log Invert Moneyness Interpolation Scale with Linear Extrapolation
plt.figure(figsize=(14,6))
plt.suptitle('Adapted Log Invert Moneyness Interpolation Scale with Linear Extrapolation',fontsize=16, fontweight='bold')
## Adapted Delta vs Spot
plt.subplot(121)
plt.plot(Sx, Delta_XIntrpL,'.-')
plt.title('Adapted Delta vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Delta',color='black')
## Adapted Gamma vs Spot
plt.subplot(122)
plt.plot(Sx, Gamma_XIntrpL,'.-')
plt.title('Adapted Gamma vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Gamma',color='black')
plt.show()



## Adapted Log Invert Moneyness Interpolation Scale with Wing Ratio Extrapolation
plt.figure(figsize=(14,6))
plt.suptitle('Adapted Log Invert Moneyness Interpolation Scale with Wing Ratio Extrapolation',fontsize=16, fontweight='bold')
## Adapted Delta vs Spot
plt.subplot(121)
plt.plot(Sx, Delta_XIntrpW,'.-')
plt.title('Adapted Delta vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Delta',color='black')
## Adapted Gamma vs Spot
plt.subplot(122)
plt.plot(Sx, Gamma_XIntrpW,'.-')
plt.title('Adapted Gamma vs Spot (Strike='+str(K)+')')
plt.xlabel('Spot',color='black')
plt.ylabel('Adapted Gamma',color='black')
plt.show()


