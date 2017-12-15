from __future__ import division
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
# import from my own packages
from SubFunctions.OptionPricing import StrikeSolver, ATMStrike, OptionPrice, OptionDelta
from SubFunctions.BrokerToSmile import BrokerToSmile
from SubFunctions.CubicSpline import CubSpline_LinExt_DeltaIntrp, CubSpline_LogMon9P, CubSpline_LogMon5P_Flat, CubSpline_LogMon5P_Linear, CubSpline_Strike_Linear
from SubFunctions.ReadData import ReadData

## 1. Read Data
[ccypair, S, MatLabel, Te, Td, Rd, Rf, 
 VAtm, RR25, BF25, RR10, BF10, 
 ATMMode, Premium, DeltaMode, StrangleMode] = ReadData()

## 2. Market Data Pre-Processing
F = S * np.exp( (Rd - Rf) * Td )
DFf = np.exp(-Rf * Te)
DFd = np.exp(-Rd * Te)
plt.style.use('ggplot')

## 3. ATM Delta & Strike
'''
Assume ATM point is defined by At-the-money Straddle delta = 0, 
and market quote delta is forward delta, premium included.
'''
DeltaAtm = OptionDelta(-1, S, F, VAtm, Rd, Rf, Te, Td, Premium, DeltaMode)
print('Premium included fwd delta = {}'.format(DeltaAtm))
KAtm     = ATMStrike(S, VAtm, Rd, Rf, Te, Td, Premium, ATMMode )
print('ATM Strike = {}'.format(KAtm))
print('\n')

## 4. 1-vol quote to 2-vol quote convertion
if StrangleMode == '1vol':
    [K25c1v, K25p1v, V25c1v, V25p1v, K25c2v, K25p2v, V25c2v, V25p2v] = BrokerToSmile(0.25, VAtm, RR25, BF25, S, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    print([K25c1v, K25p1v, V25c1v, V25p1v, K25c2v, K25p2v, V25c2v, V25p2v])
    print('K25c1v = {0}, V25c1v = {1}.'.format(K25c1v, V25c1v))
    print('K25p1v = {0}, V25p1v = {1}.'.format(K25p1v, V25p1v))
    print('K25c2v = {0}, V25c2v = {1}.'.format(K25c2v, V25c2v))
    print('K25p2v = {0}, V25p2v = {1}.'.format(K25p2v, V25p2v))
    print('\n')

    [K10c1v, K10p1v, V10c1v, V10p1v, K10c2v, K10p2v, V10c2v, V10p2v] = BrokerToSmile(0.10, VAtm, RR10, BF10, S, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    print('K10c1v = {0}, V10c1v = {1}.'.format(K10c1v, V10c1v))
    print('K10p1v = {0}, V10p1v = {1}.'.format(K10p1v, V10p1v))
    print('K10c2v = {0}, V10c2v = {1}.'.format(K10c2v, V10c2v))
    print('K10p2v = {0}, V10p2v = {1}.'.format(K10p2v, V10p2v))
    print('\n')
elif StrangleMode == '2vol':
    V25c2v = VAtm + BF25 + RR25 / 2
    V25p2v = VAtm + BF25 - RR25 / 2
    V10c2v = VAtm + BF10 + RR10 / 2
    V10p2v = VAtm + BF10 - RR10 / 2
    K25c2v = StrikeSolver( 0.25,  1, S, V25c2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    K25p2v = StrikeSolver(-0.25, -1, S, V25p2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    K10c2v = StrikeSolver( 0.10,  1, S, V10c2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    K10p2v = StrikeSolver(-0.10, -1, S, V10p2v, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode)
    print('K25c2v = {0}, V25c2v = {1}.'.format(K25c2v, V25c2v))
    print('K25p2v = {0}, V25p2v = {1}.'.format(K25p2v, V25p2v))
    print('K10c2v = {0}, V10c2v = {1}.'.format(K10c2v, V10c2v))
    print('K10p2v = {0}, V10p2v = {1}.'.format(K10p2v, V10p2v))

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
print('K1p = {0}, V1p = {1}.'.format(K1p, V1p))
print('K001p = {0}, V001p = {1}.'.format(K001p, V001p))
print('K1c = {0}, V1c = {1}.'.format(K1c, V1c))
print('K001c = {0}, V001c = {1}.'.format(K001c, V001c))



########################################### Interpolation ################################################

## 6. Interpolated Delta Points
D_DIntrp = np.linspace(-0.01,-0.99,99,endpoint=True)
N_DIntrp = len(D_DIntrp)

## 6-a. Delta Intrp
ND = 5
VQuotes5 = np.array([V10p2v, V25p2v, VAtm, V25c2v, V10c2v])
DeltaQuotes5 = np.array([-10, -25, DeltaAtm*100, 25, 10]) / 100
KQuotes5 = np.array([K10p2v, K25p2v, KAtm, K25c2v, K10c2v])
PutDeltaQuotes5 = OptionDelta(-1, S, KQuotes5, VQuotes5, Rd, Rf, Te, Td, 'Excluded', DeltaMode )
V_DIntrp, a1, b1, c1, d1 = CubSpline_LinExt_DeltaIntrp(D_DIntrp, PutDeltaQuotes5, VQuotes5 )     # Cubic Spline
K_DIntrp = np.zeros(N_DIntrp)
for i in range(N_DIntrp):
    K_DIntrp[i] = StrikeSolver(D_DIntrp[i], -1, S, V_DIntrp[i], Rd, Rf, Te, Td, ATMMode, 'Excluded', DeltaMode)  # Solve Strike
PremC_DIntrp = OptionPrice( 1, S, K_DIntrp, V_DIntrp,  Rd, Rf, Te, Td)   # Call Premium
PremP_DIntrp = OptionPrice(-1, S, K_DIntrp, V_DIntrp, Rd, Rf, Te, Td)   # Put Premium

## 6-b. Log Moneyness-9 Point Intrp
NX = 9
VQuotes9 = np.array([V001p, V1p, V10p2v, V25p2v, VAtm, V25c2v, V10c2v, V1c, V001c])
DeltaQuotes9 = np.array([-0.01, -1, -10, -25, DeltaAtm*100, 25, 10, 1, 0.01]) / 100
KQuotes9 = np.array([K001p, K1p, K10p2v, K25p2v, KAtm, K25c2v, K10c2v, K1c, K001c])
PutDeltaQuotes9 = OptionDelta(-1, S, KQuotes9, VQuotes9, Rd, Rf, Te, Td, 'Excluded', DeltaMode )
XQuotes9 = np.log(F / KQuotes9) / (VAtm**2 * Te)
K_XIntrp = np.linspace(K1p, K1c, 500, endpoint=True)
N_XIntrp = len(K_XIntrp)
X_XIntrp9 = np.log(F / K_XIntrp) / (VAtm**2 * Te)
V_XIntrp9, a2, b2, c2, d2 = CubSpline_LogMon9P(X_XIntrp9, XQuotes9, VQuotes9 )
D_XIntrp9 = OptionDelta(-1, S, K_XIntrp, V_XIntrp9, Rd, Rf, Te, Td, 'Excluded', DeltaMode )         # Delta
PremC_XIntrp9 = OptionPrice( 1, S, K_XIntrp, V_XIntrp9,  Rd, Rf, Te, Td)   # Call Premium
PremP_XIntrp9 = OptionPrice(-1, S, K_XIntrp, V_XIntrp9, Rd, Rf, Te, Td)   # Put Premium

## 6-c. Log Moneyness-5 Point Flat
XQuotes5 = np.log(F / KQuotes5) / (VAtm**2 * Te)
#K_XIntrp = np.linspace(K001p, K001c, 200, endpoint=True)
X_XIntrp5F = np.log(F / K_XIntrp) / (VAtm**2 * Te)
V_XIntrp5F, a3, b3, c3, d3 = CubSpline_LogMon5P_Flat(X_XIntrp5F, XQuotes5, VQuotes5 )
D_XIntrp5F = OptionDelta(-1, S, K_XIntrp, V_XIntrp5F, Rd, Rf, Te, Td, 'Excluded', DeltaMode )          # Delta
PremC_XIntrp5F = OptionPrice( 1, S, K_XIntrp, V_XIntrp5F,  Rd, Rf, Te, Td)   # Call Premium
PremP_XIntrp5F = OptionPrice(-1, S, K_XIntrp, V_XIntrp5F, Rd, Rf, Te, Td)   # Put Premium

## 6-d. Log Moneyness-5 Point Linear
#K_XIntrp = np.linspace(K001p, K001c, 200, endpoint=True)
X_XIntrp5L = np.log(F / K_XIntrp) / (VAtm**2 * Te)
V_XIntrp5L, a4, b4, c4, d4 = CubSpline_LogMon5P_Linear(X_XIntrp5L, XQuotes5, VQuotes5 )
D_XIntrp5L = OptionDelta(-1, S, K_XIntrp, V_XIntrp5L, Rd, Rf, Te, Td, 'Excluded', DeltaMode )          # Delta
PremC_XIntrp5L = OptionPrice( 1, S, K_XIntrp, V_XIntrp5L,  Rd, Rf, Te, Td)   # Call Premium
PremP_XIntrp5L = OptionPrice(-1, S, K_XIntrp, V_XIntrp5L, Rd, Rf, Te, Td)   # Put Premium

## 6-e. Strike-5 Point Linear
V_KIntrp, a5, b5, c5, d5 = CubSpline_Strike_Linear(K_XIntrp, KQuotes5, VQuotes5 )
D_KIntrp = OptionDelta(-1, S, K_XIntrp, V_KIntrp, Rd, Rf, Te, Td, 'Excluded', DeltaMode )          # Delta
PremC_KIntrp = OptionPrice( 1, S, K_XIntrp, V_KIntrp,  Rd, Rf, Te, Td)   # Call Premium
PremP_KIntrp = OptionPrice(-1, S, K_XIntrp, V_KIntrp, Rd, Rf, Te, Td)   # Put Premium



## 7-a. Volatility Arbitrage
# a. Delta Intrp
Diff_K_DIntrp = np.diff(K_DIntrp)
Diff_PremC_DIntrp = np.diff(PremC_DIntrp)
Diff_PremP_DIntrp = np.diff(PremP_DIntrp)
CallSpread_DIntrp = 1.0 + Diff_PremC_DIntrp / Diff_K_DIntrp
PutSpread_DIntrp = Diff_PremP_DIntrp / Diff_K_DIntrp
Butterfly_DIntrp = 2.0 * np.diff(CallSpread_DIntrp) / (Diff_K_DIntrp[0:N_DIntrp-2] + Diff_K_DIntrp[1:N_DIntrp-1])

# b. Log Moneyness - 9 point
Diff_K_XIntrp = np.diff(K_XIntrp)
Diff_PremC_XIntrp9 = np.diff(PremC_XIntrp9)
Diff_PremP_XIntrp9 = np.diff(PremP_XIntrp9)
CallSpread_XIntrp9 = 1.0 + Diff_PremC_XIntrp9 / Diff_K_XIntrp
PutSpread_XIntrp9 = Diff_PremP_XIntrp9 / Diff_K_XIntrp
Butterfly_XIntrp9 = 2.0 * np.diff(CallSpread_XIntrp9) / (Diff_K_XIntrp[0:N_XIntrp-2] + Diff_K_XIntrp[1:N_XIntrp-1])

# c. Log Moneyness - 5 flat
Diff_K_XIntrpF = np.diff(K_XIntrp)
Diff_PremC_XIntrp5F = np.diff(PremC_XIntrp5F)
Diff_PremP_XIntrp5F = np.diff(PremP_XIntrp5F)
CallSpread_XIntrp5F = 1.0 + Diff_PremC_XIntrp5F / Diff_K_XIntrpF
PutSpread_XIntrp5F = Diff_PremP_XIntrp5F / Diff_K_XIntrpF
Butterfly_XIntrp5F = 2.0 * np.diff(CallSpread_XIntrp5F) / (Diff_K_XIntrpF[0:N_XIntrp-2] + Diff_K_XIntrpF[1:N_XIntrp-1])

# d. Log Moneyness - 5 linear
Diff_K_XIntrpL = np.diff(K_XIntrp)
Diff_PremC_XIntrp5L = np.diff(PremC_XIntrp5L)
Diff_PremP_XIntrp5L = np.diff(PremP_XIntrp5L)
CallSpread_XIntrp5L = 1.0 + Diff_PremC_XIntrp5L / Diff_K_XIntrpL
PutSpread_XIntrp5L = Diff_PremP_XIntrp5L / Diff_K_XIntrpL
Butterfly_XIntrp5L = 2.0 * np.diff(CallSpread_XIntrp5L) / (Diff_K_XIntrpL[0:N_XIntrp-2] + Diff_K_XIntrpL[1:N_XIntrp-1])

# e. Strike - Linear
Diff_K_KIntrp = np.diff(K_XIntrp)
Diff_PremC_KIntrp = np.diff(PremC_KIntrp)
Diff_PremP_KIntrp = np.diff(PremP_KIntrp)
CallSpread_KIntrp = 1.0 + Diff_PremC_KIntrp / Diff_K_KIntrp
PutSpread_KIntrp = Diff_PremP_KIntrp / Diff_K_KIntrp
Butterfly_KIntrp = 2.0 * np.diff(CallSpread_KIntrp) / (Diff_K_KIntrp[0:N_XIntrp-2] + Diff_K_KIntrp[1:N_XIntrp-1])


############################################################## Ploting for Essay ####################################################################


## Delta Interpolation scale with Linear Extrapolation
plt.figure(figsize=(14,9))
plt.suptitle('Delta Interpolation Scale with Linear Extrapolation',fontsize=16, fontweight='bold')
## Plot 1: Vol vs Delta
plt.subplot(221)
plt.plot(D_DIntrp, V_DIntrp, label = 'Delta',linewidth=2.0)
plt.plot(PutDeltaQuotes5, VQuotes5,linestyle='', marker='o', markersize=5.0)
plt.title('Volatility vs Delta')
plt.xlabel('Delta',color='black')
plt.ylabel('Volatility',color='black')
## Plot 2: Vol vs K
plt.subplot(222)
plt.plot(K_DIntrp, V_DIntrp, linestyle='-',linewidth=2.0)
plt.plot(KQuotes5,VQuotes5,linestyle='None', marker='o', markersize=5.0)
plt.title('Volatility vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Volatility',color='black')
## Plot 3: Premium vs Strike
plt.subplot(223)
plt.plot(K_DIntrp[K_DIntrp<=KAtm], PremP_DIntrp[K_DIntrp<=KAtm], marker='.', label='Put')
plt.plot(K_DIntrp[K_DIntrp>=KAtm], PremC_DIntrp[K_DIntrp>=KAtm], marker='.', label='Call')
plt.title('Premium vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Premium',color='black')
plt.legend(loc=1 )
## Plot 4: Butterfly Arbitrage
plt.subplot(224)
plt.plot(K_DIntrp[1:N_DIntrp-1],Butterfly_DIntrp,linestyle='-',marker='.')
plt.title('Butterfly Arbitrage (Spot Density)')
plt.xlabel('Strike',color='black')
plt.ylabel('Spot Density',color='black')
plt.axhline(y=0.0,linestyle='--',color='black')
#plt.tight_layout()
plt.show()


## Adapted Log Invert Moneyness Interpolation scale with Flat Extrapolation
plt.figure(figsize=(14,9))
plt.suptitle('Adapted Log Invert Moneyness Interpolation Scale with Flat Extrapolation',fontsize=16, fontweight='bold')
## Plot 1: Vol vs Delta
plt.subplot(221)
plt.plot(D_XIntrp5F, V_XIntrp5F, label = 'Delta',linewidth=2.0)
plt.plot(PutDeltaQuotes5, VQuotes5,linestyle='', marker='o', markersize=5.0)
plt.title('Volatility vs Delta')
plt.xlabel('Delta',color='black')
plt.ylabel('Volatility',color='black')
## Plot 2: Vol vs K
plt.subplot(222)
plt.plot(K_XIntrp, V_XIntrp5F, linestyle='-',linewidth=2.0)
plt.plot(KQuotes5,VQuotes5,linestyle='None', marker='o', markersize=5.0)
plt.title('Volatility vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Volatility',color='black')
## Plot 3: Premium vs Strike
plt.subplot(223)
plt.plot(K_XIntrp[K_XIntrp<=KAtm], PremP_XIntrp5F[K_XIntrp<=KAtm], marker='.', label='Put')
plt.plot(K_XIntrp[K_XIntrp>=KAtm], PremC_XIntrp5F[K_XIntrp>=KAtm], marker='.', label='Call')
plt.title('Premium vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Premium',color='black')
plt.legend(loc=1 )
## Plot 4: Butterfly Arbitrage
plt.subplot(224)
plt.plot(K_XIntrp[1:N_XIntrp-1],Butterfly_XIntrp5F,linestyle='-',marker='.')
plt.title('Butterfly Arbitrage (Spot Density)')
plt.xlabel('Strike',color='black')
plt.ylabel('Spot Density',color='black')
plt.axhline(y=0.0,linestyle='--',color='black')
plt.show()


## Adapted Log Invert Moneyness Interpolation scale with Linear Extrapolation
plt.figure(figsize=(14,9))
plt.suptitle('Adapted Log Invert Moneyness Interpolation Scale with Linear Extrapolation',fontsize=16, fontweight='bold')
## Plot 1: Vol vs Delta
plt.subplot(221)
plt.plot(D_XIntrp5L, V_XIntrp5L, label = 'Delta',linewidth=2.0)
plt.plot(PutDeltaQuotes5, VQuotes5,linestyle='', marker='o', markersize=5.0)
plt.title('Volatility vs Delta')
plt.xlabel('Delta',color='black')
plt.ylabel('Volatility',color='black')
## Plot 2: Vol vs K
plt.subplot(222)
plt.plot(K_XIntrp, V_XIntrp5L, linestyle='-',linewidth=2.0)
plt.plot(KQuotes5,VQuotes5,linestyle='None', marker='o', markersize=5.0)
plt.title('Volatility vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Volatility',color='black')
## Plot 3: Premium vs Strike
plt.subplot(223)
plt.plot(K_XIntrp[K_XIntrp<=KAtm], PremP_XIntrp5L[K_XIntrp<=KAtm], marker='.', label='Put')
plt.plot(K_XIntrp[K_XIntrp>=KAtm], PremC_XIntrp5L[K_XIntrp>=KAtm], marker='.', label='Call')
plt.title('Premium vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Premium',color='black')
plt.legend(loc=1 )
## Plot 4: Butterfly Arbitrage
plt.subplot(224)
plt.plot(K_XIntrp[1:N_XIntrp-1],Butterfly_XIntrp5L,linestyle='-',marker='.')
plt.title('Butterfly Arbitrage (Spot Density)')
plt.xlabel('Strike',color='black')
plt.ylabel('Spot Density',color='black')
plt.axhline(y=0.0,linestyle='--',color='black')
plt.show()



## Adapted Log Invert Moneyness Interpolation scale with Wing Ratio Extrapolation
plt.figure(figsize=(14,9))
plt.suptitle('Adapted Log Invert Moneyness Interpolation Scale with Wing Ratio Extrapolation',fontsize=16, fontweight='bold')
## Plot 1: Vol vs Delta
plt.subplot(221)
plt.plot(D_XIntrp9, V_XIntrp9, label = 'Delta',linewidth=2.0)
plt.plot(PutDeltaQuotes9, VQuotes9,linestyle='', marker='o', markersize=5.0)
plt.title('Volatility vs Delta')
plt.xlabel('Delta',color='black')
plt.ylabel('Volatility',color='black')
## Plot 2: Vol vs K
plt.subplot(222)
plt.plot(K_XIntrp, V_XIntrp9, linestyle='-',linewidth=2.0)
plt.plot(KQuotes9[1:-1],VQuotes9[1:-1],linestyle='None', marker='o', markersize=5.0)
plt.title('Volatility vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Volatility',color='black')
## Plot 3: Premium vs Strike
plt.subplot(223)
plt.plot(K_XIntrp[K_XIntrp<=KAtm], PremP_XIntrp9[K_XIntrp<=KAtm], marker='.', label='Put')
plt.plot(K_XIntrp[K_XIntrp>=KAtm], PremC_XIntrp9[K_XIntrp>=KAtm], marker='.', label='Call')
plt.title('Premium vs Strike')
plt.xlabel('Strike',color='black')
plt.ylabel('Premium',color='black')
plt.legend(loc=1 )
## Plot 4: Butterfly Arbitrage
plt.subplot(224)
plt.plot(K_XIntrp[1:N_XIntrp-1],Butterfly_XIntrp9,linestyle='-',marker='.')
plt.title('Butterfly Arbitrage (Spot Density)')
plt.xlabel('Strike',color='black')
plt.ylabel('Spot Density',color='black')
plt.axhline(y=0.0,linestyle='--',color='black')
plt.show()



