import numpy as np
from tapy import Indicators
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import talib
from sklearn.cluster import KMeans
from smartmoneyconcepts import smc

def prepare_sequences(data,sequence_length=96):
    
    sequences = []
    labels = []
    
    for phase, values in data.items():
        for i in range(len(values) - sequence_length):
            sequence = values[i:i+sequence_length]
            label = values[i+sequence_length]
            sequences.append(sequence)
            labels.append(label)

    return np.array(sequences), np.array(labels)

def ProcessDataWithAllFunctions(data):
    data=addSMCData(data)
    data=addIndicators(data)
    data=setFairValueGaps(data)
    data=setHeikanishi(data)
    data=setCandleStickData(data)
    data=setCandleStickPatterns(data)
    data=addMinutesofDay(data)

    data= data.loc[:, ~data.columns.str.startswith('Unnamed')]
    
    return data

def addIndicators(data,sma=True,fractals=True):
    i=Indicators(data)
    if sma:
        i.sma(column_name="SMA",apply_to="High")
    if fractals :
        i.fractals()
    i.df["SMA"].fillna(0, inplace=True)
    i.df["fractals_high"]=i.df["fractals_high"].replace({True: 1, False: 0})
    i.df["fractals_low"]=i.df["fractals_low"].replace({True: 1, False: 0})
    return i.df

def setFairValueGaps(data):
    highs=data["High"].shift()
    lows=data["Low"].shift(-1)
    data["FairValue"]=highs-lows
    data["FairValue"].fillna(0, inplace=True)
    return data

def setHeikanishi(data):
    ha_close = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    ha_open = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    ha_high = data[['High', 'Open', 'Close']].max(axis=1)
    ha_low = data[['Low', 'Open', 'Close']].min(axis=1)
    data["HK_Open"]=ha_open
    data["HK_High"]=ha_high
    data["HK_Low"]=ha_low
    data["HK_Close"]=ha_close
    data["HK_Open"].fillna(0, inplace=True)
    return data

def setCandleStickData(data:pd.DataFrame):
    data['Body'] = data['Close'] - data['Open']
    data['UpperWick'] = data['High'] - data[['Open', 'Close']].max(axis=1)
    data['LowerWick'] = data[['Open', 'Close']].min(axis=1) - data['Low']

    # Create columns indicating whether the candlestick is bullish or bearish
    data['Bullish'] = data['Close'] > data['Open']
    data['Bearish'] = data['Close'] < data['Open']

    data["Bullish"]=data["Bullish"].replace({True: 1, False: 0})
    data["Bearish"]=data["Bullish"].replace({True: 1, False: 0})
    
    return data
def PirceDataNormalizer(dataframe):
    for column in dataframe.columns.to_list():
            scaler=MinMaxScaler()
            datanumpy=np.array([dataframe[column].to_numpy()]).T
    
            transformeddt=scaler.fit_transform(datanumpy).T[0]
     
            dataframe[column]=transformeddt
  
    return dataframe
def fiveMinAsFourHour(data):
    """
    Gets a 5 Min data set and returns a 4Hr data set
    """
    
    otherdata=data.copy()
    #print(otherdata)
    otherdata.reset_index(drop=True, inplace=True)

    data['4H_Group'] = otherdata.index.to_series().apply(lambda x:int(x  * (1/12))).to_numpy()
    # print(otherdata.index.to_series().apply(lambda x:int(x  * (1/12)) ).to_numpy())
    # print(data["4H_Group"])

    # Calculate OHLC for each 4-hour interval
    ohlc_4hr = data.groupby('4H_Group').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })

    # Drop NaN rows (if any)
    ohlc_4hr.dropna(inplace=True)

    return ohlc_4hr[["Open","High","Low","Close"]]
def PriceNormlizer(pricedata)->pd.DataFrame:
            """
            Normalizes the prices of the Forex Instrument
            """
            data=pricedata.copy()
          
            data.reset_index(inplace=True)
            data=data[["Open","High","Low","Close"]]
            scaler=MinMaxScaler()
            data=pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
            othercolumns=[]
            #print("Normalized data:\n-----------\n",data)
            for i in pricedata.columns.tolist():
                
                if i not in ["Open","High","Low","Close"]:
                      othercolumns.append(i)
  
            if len(othercolumns)==0:
                data["Datetime"]=pricedata.index

                newarray=data
                newarray.set_index("Datetime",inplace=True)
            else:
                newarray= pd.concat([pricedata[othercolumns], data], axis=1)
      
            return newarray
def setCandleStickPatterns(data):
    """
    Sets candle stick patterns
    """
    data["Doji"]=talib.CDLDOJI(data["Open"], data["High"], data["Low"], data["Close"])
    data["Englufing"]=talib.CDLENGULFING(data["Open"], data["High"], data["Low"], data["Close"])
    data["BreakAway"]=talib.CDLBREAKAWAY(data["Open"], data["High"], data["Low"], data["Close"])
    data["LongLeggedDoji"]=talib.CDLLONGLEGGEDDOJI(data["Open"], data["High"], data["Low"], data["Close"])
    data["MorningStar"]=talib.CDLMORNINGSTAR(data["Open"], data["High"], data["Low"], data["Close"])
    data["ShootingStar"]=talib.CDLSHOOTINGSTAR(data["Open"], data["High"], data["Low"], data["Close"])
    return data

def clusterPrice(data):

    num_clusters = 16

    # Apply K-means clustering
    highprices=data["High"]
    lowprices=data["Low"]

    highkmeans = KMeans(n_clusters=num_clusters, random_state=48)
    lowkmeans = KMeans(n_clusters=num_clusters, random_state=48)

    data['HighPriceCluster'] = highkmeans.fit_predict(highprices)
    data['LowPriceCluster'] = lowkmeans.fit_predict(lowprices)

    return data
def calculate_zigzag(data, threshold=-0.00025):
    """
    Implement Zigzag algorithm to identify turning points (peaks and troughs) in OHLC data.
    
    Args:
    - data (pandas DataFrame): DataFrame containing OHLC data.
    - threshold (float): Minimum percentage change required to form a turning point.
    
    Returns:
    - pandas Series: Zigzag line representing the turning points.
    """
    main_data=data.copy()
    data=PriceNormlizer(data)
    main_data=data.copy()
    highs = data['High']
    lows = data['Low']
    turning_points = []
    data.reset_index(drop=False,inplace=True)
    last_peak_trough = data.index[0]
    direction = None
    
    for idx in range(len(data)):
        high = highs.iloc[idx]
        low = lows.iloc[idx]
        if direction is None:
            turning_points.append((data.index[idx], high if high > low else low))
            direction = 'up' if highs.iloc[idx] < highs.iloc[last_peak_trough] else 'down'
        elif direction == 'up':
            if high > highs.iloc[last_peak_trough] * (1 + threshold):
                turning_points.append((data.index[idx], high))
                last_peak_trough = idx
                direction = 'down'
            elif low < lows.iloc[last_peak_trough]:
                last_peak_trough = idx
        elif direction == 'down':
            if low < lows.iloc[last_peak_trough] * (1 - threshold):
                turning_points.append((data.index[idx], low))
                last_peak_trough = idx
                direction = 'up'
            elif high > highs.iloc[last_peak_trough]:
                last_peak_trough = idx
                
    zigzag_line = pd.Series(index=data.index)
    for idx, price in turning_points:
        zigzag_line[idx] = price
    
    
    #data.set_index("Datetime",inplace=True)
    
    zigzag_line=zigzag_line.to_frame()

    zigzag_line["Datetime"]=pd.to_datetime(main_data['Datetime'], unit='s')
    zigzag_line.set_index("Datetime",inplace=True)
    zigzag_line = zigzag_line.interpolate(method='time')  # Fill in missing values by interpolation
    
    main_data["Zigzag"]=zigzag_line.to_numpy().T.flatten()
    main_data["Zigzag"].fillna(0)
    #print("zigzag:\n ============",data)
    #zigzag_line_df = zigzag_line.reindex(data.index).interpolate(method='time')
    return main_data
def addSMCData(data:pd.DataFrame):
    def loadcolumnAtoB(A:pd.Series,B:pd.DataFrame):
        a=A.copy()
        if type(a) is pd.Series:
            a=a.to_frame()
        b=B.copy()
        columnlist=a.columns.to_list()
        maincolumn=columnlist[0]
        for column in columnlist:
                if "Level" in column:
                    continue
                column_name=column
                if(column!=maincolumn):
                          column_name=maincolumn+"_"+column
                else:
                    pass
                b[column_name]=a[column].to_numpy()
        del A
        del B
        return b
    #print(data)
    ohlc_data=data[["Open","High","Low","Close"]].copy()
    for i in ohlc_data.columns.to_list():
        ohlc_data[i.lower()]=ohlc_data[i]
        ohlc_data=ohlc_data.drop(i,axis=1)
    fairvaluedata=smc.fvg(ohlc_data)
    fairvaluedata=fairvaluedata.fillna(-0)
    data=loadcolumnAtoB(fairvaluedata,data)
    orderblock=smc.ob(ohlc_data)
    orderblock=orderblock.fillna(-0)
    data=loadcolumnAtoB(orderblock,data)
    # print(ohlc_data)
    highs_lows=smc.highs_lows(ohlc_data)

    highs_lows=highs_lows.fillna(-0,axis=1)

    # print(highs_lows)
    highs_lows["Levels"]=highs_lows["Levels"].fillna(0)
    data=loadcolumnAtoB(highs_lows,data)
    liquidity=smc.liquidity(ohlc_data)
    
    liquidity=liquidity.fillna(-0,axis=1)
    data=loadcolumnAtoB(liquidity,data)
    return data
def addMinutesofDay(data:pd.DataFrame):
    import datetime
    def readtime(time):
        time_object = datetime.datetime.strptime(time, "%H:%M:%S")

        # Extract hours, minutes, and seconds
        hours, minutes, seconds = time_object.hour, time_object.minute, time_object.second

        # Calculate total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
    
        return total_seconds
    data["TimeofDay"]=data['Time'].apply(readtime)
    data=data.drop("Time",axis=1)

    return data
     