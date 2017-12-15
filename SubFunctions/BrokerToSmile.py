# Import packages
import numpy as np
import pandas as pd 
import scipy.optimize as opt 
import matplotlib.pyplot as plt
from scipy.stats import norm
# Define specific math operation
from numpy import exp as exp
from math import sqrt as sqrt
# import from my own packages
from SubFunctions.OptionPricing import OptionPrice, OptionDelta
from SubFunctions.StranglePrice2Vol import K2v_opt_newton, StranglePrice2Vol


def BrokerToSmile(DeltaConv, VAtm, RR, BF1v, S, Rd, Rf, Te, Td, ATMMode, Premium, DeltaMode):

    # Import functions
    Price = lambda CPSign, K, V: OptionPrice( CPSign, S, K, V, Rd, Rf, Te, Td)
    Delta = lambda CPSign, K, V: OptionDelta(CPSign, S, K, V, Rd, Rf, Te, Td, Premium, DeltaMode)

    # Solve for Call/Put K for 1-vol Strangle
    STG1v = VAtm + BF1v
    DeltaDiff_Call = lambda K: Delta( 1, K, STG1v) - DeltaConv
    DeltaDiff_Put  = lambda K: Delta(-1, K, STG1v) + DeltaConv
    Kc1v = opt.newton(DeltaDiff_Call, S,tol = 1e-8)
    Kp1v = opt.newton(DeltaDiff_Put, S, tol = 1e-8)
    PriceCall1v = Price( 1, Kc1v, STG1v)
    PricePut1v  = Price(-1, Kp1v, STG1v)
    PriceSTG1v  = PriceCall1v + PricePut1v

    # While loop to solve for K, vol for 2-vol Strangle
    BF2v_a = BF1v               # initial 1st guess
    BF2v_b = BF1v * 1.01        # initial 2nd guess
    count = 0
    epsilon = 1e-10
    BF2v_temp = 0
    BFSet = np.array([])
    FuncSet = np.array([])

    # The Secant Method to solve 2-vol Butterfly
    while (count < 100) and (abs(BF2v_a - BF2v_b) >= epsilon):
        
        PriceSTG2v_a = StranglePrice2Vol(DeltaConv, VAtm, RR, BF2v_a, S, Rd, Rf, Te, Td, Premium, DeltaMode)
        PriceSTG2v_b = StranglePrice2Vol(DeltaConv, VAtm, RR, BF2v_b, S, Rd, Rf, Te, Td, Premium, DeltaMode)
        Func_a       = PriceSTG2v_a - PriceSTG1v
        Func_b       = PriceSTG2v_b - PriceSTG1v
        SecantFunc   = ( Func_a - Func_b ) / ( BF2v_a - BF2v_b )
        BF2v_temp    = BF2v_b - Func_b / SecantFunc
        BF2v_a = BF2v_b
        BF2v_b = BF2v_temp
        BFSet = np.append(BFSet, BF2v_b)
        FuncSet = np.append(FuncSet, Func_b)
        count = count + 1

    BF2v = BF2v_b


    # Solve out the call/put K, v of 2-vol strangle by broyden1 method
    [Kc2v, Vc2v, Kp2v, Vp2v] = K2v_opt_newton(DeltaConv, VAtm, RR, BF2v, S, Rd, Rf, Te, Td, Premium, DeltaMode)
    return Kc1v, Kp1v, STG1v, STG1v, Kc2v, Kp2v, Vc2v, Vp2v



   

    

    



