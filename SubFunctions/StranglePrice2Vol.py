# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
# import from my files
from SubFunctions.OptionPricing import OptionPrice, OptionDelta

def K2v_opt_newton(DeltaConv, VAtm, RR, BF2v, S, Rd, Rf, Te, Td, Premium, DeltaMode):
    #Import functions
    Delta = lambda CPSign, K, V: OptionDelta(CPSign, S, K, V, Rd, Rf, Te, Td, Premium, DeltaMode)

    F = S * np.exp( (Rd-Rf) * Td )
    Vc2v = VAtm + BF2v + RR / 2
    Vp2v = Vc2v - RR
    DeltaDiff_Call = lambda K: Delta( 1, K, Vc2v) - DeltaConv
    DeltaDiff_Put  = lambda K: Delta(-1, K, Vp2v) + DeltaConv


    Kc2v = opt.newton(DeltaDiff_Call, F, tol = 1e-8)
    Kp2v = opt.newton(DeltaDiff_Put, F, tol = 1e-8)
    return Kc2v, Vc2v, Kp2v, Vp2v

def StranglePrice2Vol(DeltaConv, VAtm, RR, BF2v, S, Rd, Rf, Te, Td, Premium, DeltaMode):
    
    # Import functions
    Price = lambda CPSign, K, V: OptionPrice( CPSign, S, K, V, Rd, Rf, Te, Td)

    # Solve out the call/put K, v of 2-vol strangle by broyden1 method
    [Kc2v, Vc2v, Kp2v, Vp2v] = K2v_opt_newton(DeltaConv, VAtm, RR, BF2v, S, Rd, Rf, Te, Td, Premium, DeltaMode)
    
    PriceSTG2v = Price(1, Kc2v, Vc2v) + Price(-1, Kp2v, Vp2v)
    return PriceSTG2v