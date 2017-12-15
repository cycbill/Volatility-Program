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

def StrikeSolver(DeltaConv, CPSign, S, V, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode):
    Delta = lambda K: OptionDelta(CPSign, S, K, V, Rd, Rf, Te, Td, Premium, DeltaMode)
    DeltaDiff = lambda K: Delta(K) - DeltaConv
    F = S * np.exp( (Rd-Rf) * Td )
   # KAxis = np.linspace( F*0.01, 0.6, 100 )
  #  DDAxis = DeltaDiff(KAxis)
   # plt.plot(KAxis, DDAxis)
  #  plt.axhline(y=0, xmin=0, xmax=0.6)
   # plt.show()

    #KAtm = ATMStrike(S, V, Rd, Rf, Te, Td, Premium, ATMMode )
    result = opt.newton(DeltaDiff, F, tol = 1e-8)

    return result

def ATMStrike(S, V, Rd, Rf, Te, Td, Premium, ATMMode ):

    F = S * np.exp( (Rd-Rf) * Td )

    if ATMMode == 'DNS':
        if Premium == 'Excluded':
            result = F * np.exp( 0.5 * V**2 * Te)
        elif Premium == 'Included':
            result = F * np.exp(-0.5 * V**2 * Te)
    elif ATMMode == 'ATMFwd':
        result = F
    
    return result
