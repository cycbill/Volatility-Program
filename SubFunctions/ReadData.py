from __future__ import division
import numpy as np
import pandas as pd
import sys

def ReadData():
    
    ## User to input a currency pair
    ccypair = input('Please input the currency \'EURUSD\' or \'USDJPY\':')
    ccypair = ccypair.upper()
    if ccypair == 'EURUSD':
        ## EUR/USD - 3M
        S = 1.0749
        MatLabel = '3M'
        Rf = -1.161 / 100    # EUR
        Rd = 0.513 / 100     # USD
        Te = 92 / 365
        Td = 92 / 365

        VAtm = 9.71 / 100
        RR25 = -1.065 / 100
        BF25 = 0.275 / 100
        RR10 = -1.955 / 100
        BF10 = 0.855 / 100

        ATMMode = 'DNS'
        Premium = 'Excluded'
        DeltaMode = 'Spot'
        Strangle = '2vol'
    elif ccypair == 'USDJPY':
        ## USD/JPY - 3Y
        S = 107.18
        MatLabel = '3Y'
        Rf = 1.09 / 100    # USD
        Rd = -1.07 / 100     # JPY
        Te = 1095 / 365
        Td = 1097 / 365

        VAtm = 12.038 / 100
        RR25 = -1.438 / 100
        BF25 = 0.762 / 100
        RR10 = -2.73 / 100
        BF10 = 2.632 / 100
        ATMMode = 'DNS'
        Premium = 'Included'
        DeltaMode = 'Fwd'
        Strangle = '2vol'
    else:
        print('The inputed currency pair is neither EURUSD or USDJPY. Program exits.')
        sys.exit()

    return ccypair, S, MatLabel, Te, Td, Rd, Rf, VAtm, RR25, BF25, RR10, BF10, ATMMode, Premium, DeltaMode, Strangle