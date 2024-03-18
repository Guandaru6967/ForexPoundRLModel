import numpy  as np
import pandas as pd
import mplfinance as mplf
from priceprocessor import  ProcessDataWithAllFunctions,PriceNormlizer
import matplotlib.pyplot as plt
import os
class TradingBot:
        def enter_trade(self,option,tp,sl):
                pass
        def set_parameters(self,max_trades,average_tpsize,average_slsize,lot_size_max):
                pass
        def timestep(self):
                pass
        def memory_reset(self):
                pass
        def __init__(self,data):
                self.open_trades=[]
                self.BoSBuffer=[]
                self.ChoChBuffer=[]
                pass

def datadynamicprocess(df):
        import pandas as pd

        dataframe=pd.DataFrame()
        
        dataframe[[ i.title().replace("<","").replace(">","")  for i in df.columns.to_list()]]=df[[i for i in df.columns.to_list()]]
        
        

        dataframe["Date"]=pd.to_datetime(dataframe["Date"],format='%Y.%m.%d')

        dataframe.set_index("Date",inplace=True)
        dataframe=dataframe.drop("Vol",axis=1)
        # dataframe["volume"]=dataframe["Tickvol"]
        # dataframe=dataframe.drop("Tikvol",axis=1)
        dataframe=dataframe.dropna()
        

        return dataframe
def viewdata(dataframe):
        mplf.plot(dataframe, type='candle', style='charles', title='OHLC Chart', ylabel='Price')
        mplf.show()
def analysis():
        datapath=os.path.join("data/GBPUSD5MIN","GBPUSD_M5_2020_01_06_0000_2023_09_04_0045.csv")
        datasize=4000
        dataframe=datadynamicprocess(pd.read_csv(datapath)[:datasize])

        #viewdata(dataframe)

        dataframe=ProcessDataWithAllFunctions(dataframe)
        print(dataframe.columns.to_list())
        import time
        for column in dataframe.columns.to_list():
                columndata=dataframe[column].to_numpy()
                print(column.title()," : ",np.unique(columndata))
                # try:
                #         plt.plot(dataframe.index,columndata,label=column)
                #         plt.xlabel(dataframe.index.name)
                #         plt.ylabel(column)
                #         plt.show(block=False)
                #         time.sleep(10.00)
                #         plt.close()
                # except Exception as E:
                #         print(E)
if __name__=="__main__":
        analysis()