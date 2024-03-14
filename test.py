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
from priceprocessor import addMinutesofDay,ProcessDataWithAllFunctions
import torch
import tradermade as tm
import requests
from torchrl.envs import (TransformedEnv,Compose,DoubleToFloat,StepCounter,ObservationTransform)
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import (SamplerWithoutReplacement)

from tensordict.nn.distributions import NormalParamExtractor
from trading_algo import GBPForexEnvironment
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    print(f"Number of available GPUs: {num_gpus}")

    # Access information about each GPU
    for gpu_id in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"GPU {gpu_id}: {gpu_name}")
else:
    print("No GPU available. PyTorch will use CPU.")


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
datapath=os.path.join("data/GBPUSD5MIN","GBPUSD_M5_2020_01_06_0000_2023_09_04_0045.csv")
df=pd.read_csv(datapath,delimiter="\t")
def datadynamicprocess(df):

        dataframe=pd.DataFrame()

        dataframe[[ i.title().replace("<","").replace(">","")  for i in df.columns.to_list()]]=df[[i for i in df.columns.to_list()]]


        dataframe=ProcessDataWithAllFunctions(dataframe[0:10000])

        dataframe["Date"]=pd.to_datetime(dataframe["Date"],format='%Y.%m.%d')

        dataframe.set_index("Date",inplace=True)
        dataframe=dataframe.drop("Vol",axis=1)
        dataframe=dataframe.drop("Time",axis=1)
        return dataframe
df.to_csv(datapath)
quit()
base_env=GBPForexEnvironment(datadynamicprocess(df),account_balance=100)
base_env=TestEnv()
env=TransformedEnv(base_env,Compose(DoubleToFloat(),StepCounter()))
print(env.action_spec)
quit()
df=generateSineTestData()
#print(data)
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
mplf.plot(df[1018:], type='candle', style='charles', title='OHLC Chart', ylabel='Price')
mplf.show()
# dict of functions by group
# for group, names in talib.get_function_groups():
#     print(group)
#     for name in names:
#         print(name)
def step(self,action):
                """
                A forex trading action can either be a 
                        
                        
                        0.buy -set profit and loss in pips
                        1.sell -set profit and loss in pips
                        2.modify tp /sl (trailing stop loss) 
                        3.wait 
                        
                of the form 

                action=Dict({"Entry":0.0,LotSize":0.25,"Option":Discrete(2),"TakeProfit":Box(0,70000,(1,1)),"StopLoss":Box(0,70000,(1,1))})

                A sell option is 0 while a buy option is 1 , -1 is modify tp
                """
                print("======Action======\n",action)
                truncated=False
                #self.action=Dict({"LotSize":Box(0,70000,(1,1)),"Option":Discrete(3),"TakeProfit":Box(0,70000,(1,1)),"StopLoss":Box(0,70000,(1,1))})
                
               
                action[0]=self.trade_index
                print("Made Action:",action)
                
                done=False
                
                trades_to_close=[]
                outcome=0
                reward=0
                
                #Actions Evalution
                for trade in self.open_trades:
                        
                        current_high=self.currencies.iloc[self.current_price]["High"]
                        current_low=self.currencies.iloc[self.current_price]["Low"]
                        print("Trade Entry:",trade_entry,current_high)
                        buy_trade_result=self.calculate_pips(trade_entry,current_high)
                        sell_trade_result=self.calculate_pips(current_low,trade_entry)

                        buy_trade_stop=self.calculate_pips(current_low,trade_entry)
                        sell_trade_stop=self.calculate_pips(trade_entry,current_high)

                       

                        outcome=0 # profit or loss
                        trade_entry=trade[3]
                        trade_stop_loss=trade[5][0]
                        trade_take_profit=trade[4][0]
                        trade_lot_size=trade[1]

                        trade_id=trade[0]
                        trade_option=trade[2]
                        # if trade is a buy
                        if trade_option==0:
                                #Hit buy stop loss
                                if abs(trade_stop_loss)<=abs(buy_trade_stop):
                                        if buy_trade_stop>0:
                                                reward=-trade_stop_loss
                                                trades_to_close.append(trade)
                                                outcome-=reward*trade_lot_size*10
                                                print(" buy trade stop is given by:", current_low,trade_entry)
                                                print("Stop loss diff: ",trade_stop_loss,buy_trade_stop)
                                                print(f"Buy trade trade {trade_id} has hit stop loss at {current_low} from entry {trade_entry}")
                                                print("Outcome:",outcome)
                                #Hit sell take profit
                                if trade_take_profit<=buy_trade_result:
                                        reward+=trade_take_profit
                                        trades_to_close.append(trade)
                                        outcome+=reward*trade_lot_size*10
                                        print(" buy trade result is given by:", current_high,trade_entry)
                                        print("Take profit diff: ",trade_take_profit,buy_trade_result)
                                        print(f"Buy trade trade {trade_id} has hit take profit at {current_high} from entry {trade_entry}")
                                        print("Outcome:",outcome)
                        #if trade is a sell
                        elif  trade_option==1:
                                if abs(trade_stop_loss)<=abs(sell_trade_stop):
                                        if sell_trade_stop>0:
                                                #if trade is in break even mode or in profit mode
                                                reward=-trade_stop_loss
                                                trades_to_close.append(trade)
                                                outcome-=reward*trade_lot_size*10
                                                print(" sell trade stop is given by:", current_low,trade_entry)
                                                print("Stop loss diff: ",trade_stop_loss,sell_trade_stop)
                                                print(f"Sell trade  {trade_id} has hit stop loss at {current_low} from entry {trade_entry}")
                                                print("Outcome:",outcome)
                                if trade_take_profit<=sell_trade_result:
                                        if(sell_trade_result)>0:
                                                reward=+trade_take_profit
                                                trades_to_close.append(trade)
                                                outcome+=reward*trade_lot_size*10
                                                
                                                print(" Sell trade result is given by:", current_low,trade_entry)
                                                print("Take profit diff: ",trade_take_profit,sell_trade_result)
                                                print(f"Sell trade  {trade_id} has hit take profit at {current_low} from entry {trade_entry}")
                                                print("Outcome:",outcome)
                        if type(outcome) is not int:
                                outcome=outcome[0]
                        self.account_balance=self.account_balance+outcome
                        
                                
                        #if in profit reward the pips

                        #if in loss remove the pips:
                print(len(trades_to_close),trades_to_close,self.open_trades)
                for i in trades_to_close:
                        self.open_trades.remove(i)   
                #Action Exectution
                TakeProfit=action[4][0]
                StopLoss=action[5][0]
                Option=action[2]

                if Option==1:
                        entry=self.currencies.iloc[self.current_price]["Low"]
                        print(action[4])
                        print(entry)
                        action[3]=entry
                        stoppips=self.calculate_pips(entry,StopLoss)
                        reward-=(self.average_loss_pips-stoppips)
                        action[5]=np.array([stoppips])
                        action[4]=np.array([self.calculate_pips(TakeProfit,entry)])
                        self.open_trades.append(action)
                        self.track_data["trades"].append(action)
                        print("Made a Sell Trade:" ,action)
                if Option==0:
                        entry=self.currencies.iloc[self.current_price]["High"]
                        action[3]=entry
                        self.track_data["trades"].append(action)
                        stoppips=self.calculate_pips(StopLoss ,entry)
                        reward-=(self.average_loss_pips-stoppips)
                        action[5]=np.array([stoppips])
                        action[4]=np.array([self.calculate_pips(entry,TakeProfit)])
                        self.open_trades.append(action)
                        print("Made a Buy Trade",action)
                if Option==2:
                        for trade in self.open_trades:
                                if trade["Id"]==action["Id"]:
                                                #if its a buy trade
                                                if trade["Option"]==0:
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss ,entry_price)
                                                        trade["TakeProfit"]=self.calculate_pips(TakeProfit,trade["Entry"])
                                                elif trade["Option"]==1: 
                                                        #trade["StopLoss"]=StopLoss
                                                        trade["TakeProfit"]=self.calculate_pips(trade["Entry"],TakeProfit)
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss,trade["Entry"])
                if Option==3:
                        reward+=1
                        pass
                                        
                #Add price to the environment
               

                
                if self.currencies.shape[0]-1==self.current_price:
                                done=True 
                if self.currencies.shape[0]-1==self.current_price-self.temporal_window:
                        done=True
              
                if Option==0 or Option==1:
                        self.trade_index+=1

                #Update render data
                
                if outcome>0:
                        self.track_data["profits"].append(outcome)
                        self.track_data["losses"].append(0)
                if outcome<0:
                        self.track_data["profits"].append(0)
                        self.track_data["losses"].append(outcome)
                if outcome==0:
                        self.track_data["profits"].append(0)
                        self.track_data["losses"].append(0)

                self.track_data["rewards"].append(reward)
                self.track_data["steps"].append(len(self.track_data["steps"]))
                self.track_data["balance"].append(self.account_balance)
                
                state={"temporal_window_state":self.currencies.iloc[self.current_price:self.current_price+self.temporal_window][self.features_fields].to_numpy(),
                                                                                       "balance":np.array([float(self.account_balance)]),
                                                                                       "equity":np.array([float(self.account_equity)]),
                                                                                       "margin":np.array([float(self.account_balance)]),
                                                                                       "spread":np.array([float(self.currency_spread)]),
                                                                                     
                                                                                       }
                info={"active-trades":len(self.open_trades),"outcomes":outcome,"closed-trades":len(trades_to_close)}
                self.current_price+=1
                return state,reward ,done,truncated,info

def dictactionstep(self,action):
                """
                A forex trading action can either be a 
                        
                        
                        0.buy -set profit and loss in pips
                        1.sell -set profit and loss in pips
                        2.modify tp /sl (trailing stop loss) 
                        3.wait 
                        
                of the form 

                action=Dict({"Entry":0.0,LotSize":0.25,"Option":Discrete(2),"TakeProfit":Box(0,70000,(1,1)),"StopLoss":Box(0,70000,(1,1))})

                A sell option is 0 while a buy option is 1 , -1 is modify tp
                """
                print("======Action======\n",action)
                truncated=False
                #self.action=Dict({"LotSize":Box(0,70000,(1,1)),"Option":Discrete(3),"TakeProfit":Box(0,70000,(1,1)),"StopLoss":Box(0,70000,(1,1))})
                
               
                action["Id"]=self.trade_index
                print("Made Action:",action)
                
                done=False
                TakeProfit=action["TakeProfit"][0]
                StopLoss=action["StopLoss"][0]
                trades_to_close=[]
                outcome=0
                reward=0
                
                #Actions Evalution
                for trade in self.open_trades:
                        entry_price=trade["Entry"]
                        current_high=self.currencies.iloc[self.current_price]["High"]
                        current_low=self.currencies.iloc[self.current_price]["Low"]
                        print("Trade Entry:",entry_price,current_high)
                        buy_trade_result=self.calculate_pips(entry_price,current_high)
                        sell_trade_result=self.calculate_pips(current_low,entry_price)

                        buy_trade_stop=self.calculate_pips(current_low,entry_price)
                        sell_trade_stop=self.calculate_pips(entry_price,current_high)

                        trade_lot_size=trade["LotSize"]

                        outcome=0 # profit or loss
                        trade_stop_loss=trade["StopLoss"][0]
                        trade_take_profit=trade["TakeProfit"][0]

                        # if trade is a buy
                        if trade["Option"]==0:
                                #Hit buy stop loss
                                if abs(trade_stop_loss)<=abs(buy_trade_stop):
                                        if buy_trade_stop>0:
                                                reward=-trade_stop_loss
                                                trades_to_close.append(trade)
                                                outcome-=reward*trade_lot_size*10
                                                print(" buy trade stop is given by:", current_low,entry_price)
                                                print("Stop loss diff: ",trade_stop_loss,buy_trade_stop)
                                                print(f"Buy trade trade {trade['Id']} has hit stop loss at {current_low} from entry {trade['Entry']}")
                                                print("Outcome:",outcome)
                                #Hit sell take profit
                                if trade_take_profit<=buy_trade_result:
                                        reward+=trade_take_profit
                                        trades_to_close.append(trade)
                                        outcome+=reward*trade_lot_size*10
                                        print(" buy trade result is given by:", current_high,entry_price)
                                        print("Take profit diff: ",trade_take_profit,buy_trade_result)
                                        print(f"Buy trade trade {trade['Id']} has hit take profit at {current_high} from entry {trade['Entry']}")
                                        print("Outcome:",outcome)
                        #if trade is a sell
                        elif  trade["Option"]==1:
                                if abs(trade_stop_loss)<=abs(sell_trade_stop):
                                        if sell_trade_stop>0:
                                                #if trade is in break even mode or in profit mode
                                                reward=-trade_stop_loss
                                                trades_to_close.append(trade)
                                                outcome-=reward*trade_lot_size*10
                                                print(" sell trade stop is given by:", current_low,entry_price)
                                                print("Stop loss diff: ",trade_stop_loss,sell_trade_stop)
                                                print(f"Sell trade  {trade['Id']} has hit stop loss at {current_low} from entry {trade['Entry']}")
                                                print("Outcome:",outcome)
                                if trade_take_profit<=sell_trade_result:
                                        if(sell_trade_result)>0:
                                                reward=+trade_take_profit
                                                trades_to_close.append(trade)
                                                outcome+=reward*trade_lot_size*10
                                                
                                                print(" Sell trade result is given by:", current_low,entry_price)
                                                print("Take profit diff: ",trade_take_profit,sell_trade_result)
                                                print(f"Sell trade  {trade['Id']} has hit take profit at {current_low} from entry {trade['Entry']}")
                                                print("Outcome:",outcome)
                        if type(outcome) is not int:
                                outcome=outcome[0]
                        self.account_balance=self.account_balance+outcome
                        
                                
                        #if in profit reward the pips

                        #if in loss remove the pips:
                print(len(trades_to_close),trades_to_close,self.open_trades)
                for i in trades_to_close:
                        self.open_trades.remove(i)   
                #Action Exectution
               
                if action["Option"]==1:
                        entry=self.currencies.iloc[self.current_price]["Low"]
                        print(action["TakeProfit"])
                        print(entry)
                        action["Entry"]=entry
                        stoppips=self.calculate_pips(entry,StopLoss)
                        reward-=(self.average_loss_pips-stoppips)
                        action["StopLoss"]=np.array([stoppips])
                        action["TakeProfit"]=np.array([self.calculate_pips(TakeProfit,entry)])
                        self.open_trades.append(action)
                        self.track_data["trades"].append(action)
                        print("Made a Sell Trade:" ,action)
                if action["Option"]==0:
                        entry=self.currencies.iloc[self.current_price]["High"]
                        action["Entry"]=entry
                        self.track_data["trades"].append(action)
                        stoppips=self.calculate_pips(StopLoss ,entry)
                        reward-=(self.average_loss_pips-stoppips)
                        action["StopLoss"]=np.array([stoppips])
                        action["TakeProfit"]=np.array([self.calculate_pips(entry,TakeProfit)])
                        self.open_trades.append(action)
                        print("Made a Buy Trade",action)
                if action["Option"]==2:
                        for trade in self.open_trades:
                                if trade["Id"]==action["Id"]:
                                                #if its a buy trade
                                                if trade["Option"]==0:
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss ,entry_price)
                                                        trade["TakeProfit"]=self.calculate_pips(TakeProfit,trade["Entry"])
                                                elif trade["Option"]==1: 
                                                        #trade["StopLoss"]=StopLoss
                                                        trade["TakeProfit"]=self.calculate_pips(trade["Entry"],TakeProfit)
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss,trade["Entry"])
                if action["Option"]==3:
                        reward+=1
                        pass
                                        
                #Add price to the environment
               

                
                if self.currencies.shape[0]-1==self.current_price:
                                done=True 
                if self.currencies.shape[0]-1==self.current_price-self.temporal_window:
                        done=True
              
                if action["Option"]==0 or action["Option"]==1:
                        self.trade_index+=1

                #Update render data
                
                if outcome>0:
                        self.track_data["profits"].append(outcome)
                        self.track_data["losses"].append(0)
                if outcome<0:
                        self.track_data["profits"].append(0)
                        self.track_data["losses"].append(outcome)
                if outcome==0:
                        self.track_data["profits"].append(0)
                        self.track_data["losses"].append(0)

                self.track_data["rewards"].append(reward)
                self.track_data["steps"].append(len(self.track_data["steps"]))
                self.track_data["balance"].append(self.account_balance)
                
                state={"temporal_window_state":self.currencies.iloc[self.current_price:self.current_price+self.temporal_window][self.features_fields].to_numpy(),
                                                                                       "balance":np.array([float(self.account_balance)]),
                                                                                       "equity":np.array([float(self.account_equity)]),
                                                                                       "margin":np.array([float(self.account_balance)]),
                                                                                       "spread":np.array([float(self.currency_spread)]),
                                                                                     
                                                                                       }
                info={"active-trades":len(self.open_trades),"outcomes":outcome,"closed-trades":len(trades_to_close)}
                self.current_price+=1
                return state,reward ,done,truncated,info
        