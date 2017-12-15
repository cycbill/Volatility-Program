# Import packages
from __future__ import division
import numpy as np
import pandas as pd
import scipy as sp 
import scipy.optimize as opt 
import matplotlib.pyplot as plt
from scipy.stats import norm


def d1(CPSign, S, K, V, Rd, Rf, Te, Td):
    F = S * np.exp( (Rd-Rf) * Td )
    result = (np.log(F/K) + 0.5 * V**2 * Te ) / (V * np.sqrt(Te))
    return result

def d2(CPSign, S, K, V, Rd, Rf, Te, Td):
    result = d1(CPSign, S, K, V, Rd, Rf, Te, Td) - V * np.sqrt(Te)
    return result

def OptionPrice(CPSign, S, K, V, Rd, Rf, Te, Td):
    F = S * np.exp( (Rd-Rf) * Td )  #spot date to delivery date
    Nd1 = norm.cdf( CPSign * d1(CPSign, S, K, V, Rd, Rf, Te, Td) )
    Nd2 = norm.cdf( CPSign * d2(CPSign, S, K, V, Rd, Rf, Te, Td) )
    result = CPSign * np.exp(-Rd * Td)* ( F * Nd1 - K * Nd2 )
    return result

def OptionDelta(CPSign, S, K, V, Rd, Rf, Te, Td, Premium, DeltaMode ):
    F = S * np.exp( (Rd-Rf) * Td )

    if Premium == 'Excluded':
        Nd1 = norm.cdf( CPSign * d1(CPSign, S, K, V, Rd, Rf, Te, Td) )
        result = CPSign * Nd1
    elif Premium == 'Included':
        Nd2 = norm.cdf( CPSign * d2(CPSign, S, K, V, Rd, Rf, Te, Td) )
        result = CPSign * K / F * Nd2

    if DeltaMode == 'Spot':
        result = result * np.exp( - Rf * Td )

    return result


S = 1.1188
K = np.linspace(0.9,1.3,100,endpoint=True)
V = 8.11 / 100
Rf = -0.515680041413905 / 100
Rd = 0.62999999999987 / 100
Te = 94 / 365
Td = 92 / 365
Prem_Excl_Delta_Put = OptionDelta(-1, S, K, V, Rd, Rf, Te, Td, 'Excluded', 'Forward' )
Prem_Incl_Delta_Put = OptionDelta(-1, S, K, V, Rd, Rf, Te, Td, 'Included', 'Forward' )

Prem_Excl_Delta_Call = OptionDelta(1, S, K, V, Rd, Rf, Te, Td, 'Excluded', 'Forward' )
Prem_Incl_Delta_Call = OptionDelta(1, S, K, V, Rd, Rf, Te, Td, 'Included', 'Forward' )


plt.style.use('ggplot')
plt.plot(K,Prem_Excl_Delta_Put,linestyle='--',linewidth=2,color='b',label='Premium Excluded Delta')
plt.plot(K,Prem_Incl_Delta_Put,linestyle='-.',linewidth=2,color='r',label='Premium Included Delta')
plt.xlim([0.9,1.3])
plt.ylim([-1.2,0.2])
plt.xlabel('Strike',fontsize=20,color='black')
plt.ylabel('Delta',fontsize=20,color='black')
plt.xticks( fontsize = 12,color='black')
plt.yticks( fontsize = 12,color='black')
plt.title('Put Delta - Premium Excluded & Included',fontsize=20,color='black')
plt.legend(loc = 1,fontsize = 14)
plt.show()

plt.plot(K,Prem_Excl_Delta_Call,linestyle='--',linewidth=2,color='b',label='Premium Excluded Delta')
plt.plot(K,Prem_Incl_Delta_Call,linestyle='-.',linewidth=2,color='r',label='Premium Included Delta')
plt.xlim([0.9,1.3])
plt.ylim([-0.2,1.2])
plt.xlabel('Strike',fontsize=20,color='black')
plt.ylabel('Delta',fontsize=20,color='black')
plt.xticks( fontsize = 12,color='black')
plt.yticks( fontsize = 12,color='black')
plt.title('Call Delta - Premium Excluded & Included',fontsize=20,color='black')
plt.legend(loc = 1,fontsize = 14)
plt.show()

S = 110.1188
K = np.linspace(10,200,100,endpoint=True)
V = 18 / 100
Rf = -0.515680041413905 / 100
Rd = 0.62999999999987 / 100
Te = 2000 / 365
Td = 2000 / 365
Prem_Excl_Delta_Put = OptionDelta(-1, S, K, V, Rd, Rf, Te, Td, 'Excluded', 'Forward' )
Prem_Incl_Delta_Put = OptionDelta(-1, S, K, V, Rd, Rf, Te, Td, 'Included', 'Forward' )

Prem_Excl_Delta_Call = OptionDelta(1, S, K, V, Rd, Rf, Te, Td, 'Excluded', 'Forward' )
Prem_Incl_Delta_Call = OptionDelta(1, S, K, V, Rd, Rf, Te, Td, 'Included', 'Forward' )

F = S*np.exp((Rd-Rf)*Te)
KAtm = F*np.exp(0.5 * V**2 * Te)
DeltaAtm = 0.5 * KAtm / F
print(DeltaAtm)

plt.plot(K,Prem_Excl_Delta_Call,linestyle='--',linewidth=2,color='b',label='Premium Excluded Delta')
plt.plot(K,Prem_Incl_Delta_Call,linestyle='-.',linewidth=2,color='r',label='Premium Included Delta')
plt.axhline(y=DeltaAtm, linewidth=1, linestyle='--',color = 'black')
plt.text(25, DeltaAtm+0.02, 'ATM Delta', fontsize=14,style='italic')
plt.xlim([10,200])
plt.ylim([-0.2,1.2])
plt.xlabel('Strike',fontsize=20,color='black')
plt.ylabel('Delta',fontsize=20,color='black')
plt.xticks( fontsize = 12,color='black')
plt.yticks( fontsize = 12,color='black')
plt.title('Call Delta - Premium Excluded & Included',fontsize=20,color='black')
plt.legend(loc = 1,fontsize = 14)
plt.show()