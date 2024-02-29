import numpy as np
import pandas as pd
import talib as talib
from talib.abstract import *
def zigzag(data,threshold):
    """
    Implements a Zigzag function to identify peaks and troughs in the data.

    Parameters:
    - data: Pandas Series or NumPy array of price data.
    - threshold: Minimum percentage change to identify a peak or trough.

    Returns:
    - Pandas Series with 1 for peaks, -1 for troughs, and 0 for other points.
    """
    highs = np.zeros_like(data)
    lows = np.zeros_like(data)

    # Find peaks
    for i in range(1, len(data)-1):
        if data[i] > data[i-1] and data[i] > data[i+1] and (data[i] - data[i-1]) / data[i-1] > threshold:
            highs[i] = 1

    # Find troughs
    for i in range(1, len(data)-1):
        if data[i] < data[i-1] and data[i] < data[i+1] and (data[i-1] - data[i]) / data[i-1] > threshold:
            lows[i] = -1

    zigzag_result = highs + lows
    return pd.Series(zigzag_result, index=data.index)
def fibonnaci(endpoint,startpoint,levels=[0,0.75,1],directionup=True):
    if(directionup):

        diff=endpoint-startpoint
    else:
        diff=startpoint-endpoint
    return (levels-diff)
    return 

