from MetaTrader5 import *
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import gym_anytrading
import gym
# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
 
# request connection status and parameters
print(mt5.terminal_info())

account=9709217
#help(mt5.login)
authorized=mt5.login(account,server="FBS-Demo",password="O/4NSLa:")  # the terminal database password is applied if connection data is set to be remembered
if authorized:
    print("connected to account #{}".format(account))
else:
    print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
help(mt5.copy_rates_from)
gbpusd_ticks = mt5.copy_rates_from("GBPUSD",mt5.TIMEFRAME_M5, datetime(2024,1,30,13), 2000)
# request ticks from AUDUSD within 2019.04.01 13:00 - 2019.04.02 13:00
#audusd_ticks = mt5.copy_ticks_range("AUDUSD", datetime(2020,1,27,13), datetime(2020,1,28,13), mt5.COPY_TICKS_ALL)
print(gbpusd_ticks)
timeframe_4hr = 240  # 4 hours in minutes

# Calculate moving average as fair value
fair_value = sum(candle['close'] for candle in gbpusd_ticks) / len(gbpusd_ticks)

# Identify gaps
threshold = 0.005  # Adjust as needed
gaps = [candle for candle in gbpusd_ticks if abs(candle['close'] - fair_value) > threshold]
columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
df = pd.DataFrame(gbpusd_ticks, columns=columns)

# Convert time to datetime format
df['time'] = pd.to_datetime(df['time'], unit='s')

# Plotting the candlestick chart
fig, ax = plt.subplots()
candlestick_ohlc(ax, zip(mdates.date2num(df['time']), df['open'], df['high'], df['low'], df['close']), width=0.6, colorup='green', colordown='red', alpha=0.8)

# Formatting
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator())

plt.title('Candlestick Chart')
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()
# Implement trading logic (e.g., print details of gap candles)
for gap in gaps:
    print(f"Gap detected at {datetime.utcfromtimestamp(gap['time'])}, Actual Close: {gap['close']}, Fair Value: {fair_value}")
# # Connect to MetaTrader 5
# mt5 = MetaTrader5(login="9709217", password="0a;)zD%J", server="FBS-Demo")
# mt5.connect()
# help(mt5)
# # Set the symbol, time frame, and the number of bars you want to retrieve
# symbol = 'GBPUSD'
# timeframe = MetaTrader5.TIMEFRAME_H1  # You can change this to other timeframes like TIMEFRAME_H1
# number_of_bars = 100  # Number of bars to retrieve

# # Retrieve historical candlestick data
# candlestick_data = mt5.copy_rates(symbol, timeframe, 0, number_of_bars)

# # Display the retrieved data
# for candle in candlestick_data:
#     print(f"Time: {candle['time']}, Open: {candle['open']}, High: {candle['high']}, Low: {candle['low']}, Close: {candle['close']}, Volume: {candle['tick_volume']}")

# # Disconnect from MetaTrader 5
# mt5.disconnect()