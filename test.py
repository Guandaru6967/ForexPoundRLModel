import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import datetime
import mplfinance as mplf
from tapy.indicators import Indicators

import multiprocessing
import matplotlib
import gymnasium as gym
from PIL import Image as PILImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import datetime
from priceprocessor import addMinutesofDay
from kivy.uix.floatlayout import*
from kivy.app import App


def generateSineTestData():
        start_price = 1.30000  # Starting price
        price_range = (1.10000, 1.50000)  # Price range
        num_intervals = 240 * 12  # 240 hours * 12 intervals per hour (5-minute intervals)
        swing_high_interval = 12  # Approximate interval between swing highs (1 hour = 12 intervals)
        swing_high_amplitude = 0.0020  # 20 pips amplitude

        # Generate time indices
        time_indices = np.arange(num_intervals)

        # Generate sine wave pattern
        sine_wave = np.sin(time_indices / swing_high_interval) * swing_high_amplitude

        # Generate OHLC data
        ohlc_data = []
        current_price = start_price

        start_price = 1.30000  # Starting price
        price_range = (1.10000, 1.50000)  # Price range
        num_intervals = 240 * 12  # 240 hours * 12 intervals per hour (5-minute intervals)
        swing_high_interval = 12  # Approximate interval between swing highs (1 hour = 12 intervals)
        swing_high_amplitude = 0.0020  # 20 pips amplitude

        # Generate time indices
        time_indices = np.arange(num_intervals)

        # Generate sine wave pattern
        sine_wave = np.sin(time_indices / swing_high_interval) * swing_high_amplitude

        # Generate OHLC data
        ohlc_data = []
        current_price = start_price

        for i in range(num_intervals):
                # Calculate swing high
                if i % swing_high_interval == 0:
                        swing_high = current_price + sine_wave[i]
                else:
                        swing_high = None
                
                # Update current price
                if swing_high is not None:
                        current_price = swing_high
                else:
                        current_price += sine_wave[i]
                
                # Ensure price stays within range
                current_price = max(price_range[0], min(price_range[1], current_price))
                
                # Calculate open, high, low, and close prices
                open_price = current_price if swing_high is not None else ohlc_data[-1][3]  # Use previous low as open if no swing high
                high_price = max(current_price, swing_high) if swing_high is not None else current_price
                low_price = min(current_price, swing_high) if swing_high is not None else current_price
                close_price = current_price
                
                # Append OHLC data
                ohlc_data.append([open_price, high_price, low_price, close_price])

        # Create DataFrame
        # Create DataFrame
        columns = ['Open' ,'High', 'Low', 'Close']
        index = pd.date_range(start='2024-01-01', periods=num_intervals, freq='5T')
        
        df = pd.DataFrame(ohlc_data, columns=columns, index=index)
        #print(df)
        #df=df[['Open' ,'High', 'Low', 'Close']]
        return df
df=pd.read_csv(os.path.join("data","SINE_FXENV_TESTDATA.csv"))
data=addMinutesofDay(df)["TimeofDay"].to_numpy()[24:100]
df=generateSineTestData()
print(data)
# df["Datetime"]=df["Unnamed: 0"]
# df=df.drop("Unnamed: 0",axis=1)
# df.set_index("Datetime",drop=True,inplace=True)
# print(df)
#array=gym.spaces.Sequence(gym.spaces.Dict({"Id":gym.spaces.Discrete(4),"Entry":gym.spaces.Box(high=1.5,low=1.25,dtype=np.float64)}))


# print(datatest)
# print(array.contains(datatest[0]))
# print(isinstance(datatest,tuple))
# df=generateSineTestData()
# #matplotlib.use("GTK3Agg")
mplf.plot(df[2000:], type='candle', style='charles', title='OHLC Chart', ylabel='Price')
mplf.show()
# dict of functions by group
# for group, names in talib.get_function_groups():
#     print(group)
#     for name in names:
#         print(name)