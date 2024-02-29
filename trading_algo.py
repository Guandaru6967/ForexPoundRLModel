from matplotlib.pylab import Generator
from  stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
# from keras.layers import Activation,MaxPool3D,AvgPool3D,ConvLSTM3D,LSTM,Dense
# from keras.activations import linear,relu,softmax,sigmoid
# from keras.optimizers import Adam,Adamax 
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from gym.spaces import Box,MultiDiscrete,MultiBinary,Dict,Discrete
import gymnasium as gym
import uuid
import pandas as pd
import os

from priceprocessor import PriceNormlizer
class ObservationSpace(gym.Space):
        def __init__(self, *args,**kwargs):
                super().__init__(*args,**kwargs)
                print(kwargs)
                print(args)

class GBPForexEnvironment(gym.Env):
        def __init__(self,data:dict,*args,account_balance=100,risk_size=0.3,currency_spreads={"GBPUSD":0},**kwargs) -> None:
                """
                data: 
                {
                        "GBPUSD": pd.DataFrame()
                
                }
                """
                super().__init__(*args,**kwargs)
                self.currencies={}
                self.max_price=0.0
                self.min_price=0.00
                count=0
                
                self.current_price=0
                
                for i in data.keys():
                        self.currencies[count]=data[i]#PriceNormlizer(data[i])
                        self.max_price=max([self.max_price,data[i]["High"].max()])
                        self.min_price=min([self.min_price, data[i]["Low"].min()])
                        count+=1
                
                #Account Management settings
                self.account_balance=account_balance
                self.__default_account_balance__=self.account_balance
                self.account_equity=self.__default_account_balance__
                self.margin=self.account_balance
                self.lot_size_max=(risk_size*self.account_balance)/(20*self.calculate_pip_precision(self.max_price))
                print("Max Lot size: ", self.lot_size_max)
                
                self.lot_ratio=0
                self.risk_reward_ratio=2.0
                self.open_trades=[]
                self.currency_spreads=currency_spreads
                self.max_trades=10
                self.trade_index=0
                #Trade Execution Spaces
                self.lost_size_space=gym.spaces.Box(low=0.01,high=self.lot_size_max)
                self.stop_loss_space=gym.spaces.Box(low=float(self.min_price),high=float(self.max_price))
                self.trade_option_space=gym.spaces.Discrete(3)
                self.trade_entry_space=gym.spaces.Box(low=float(self.min_price),high=float(self.max_price))
                self.take_profit_space=gym.spaces.Box(high=float(self.max_price),low=float(self.min_price))
                #Class spaces
                if len(data.keys())==0:
                        maxpair=1
                else:
                        maxpair=len(data.keys())
                self.action_space=gym.spaces.Dict({"Id":gym.spaces.Discrete(self.max_trades),"pair":gym.spaces.Discrete(maxpair),"LotSize":self.lost_size_space,"Option":self.trade_option_space,"Entry":self.trade_entry_space,"TakeProfit":self.take_profit_space,"StopLoss":self.stop_loss_space})
                print("sample:",self.action_space.sample())
                self.observation_space =gym.spaces.Box(low=0.0,high=1000000.0,shape=(3,),dtype=np.float64)#Dict({"balance":Box(0,100000000,shape=(1,)),"equity":Box(0,100000000,shape=(1,)),"margin":Box(0,100000000,shape=(1,))})  
                print(isinstance(self.action_space,gym.spaces.Dict))
                print("Observation:",self.observation_space.sample())
                
        
        def calculate_pips(self,price1, price2):
                """
                Calculate the number of pips between two prices.

                Parameters:
                - price1: The first price.
                - price2: The second price.
                - pip_precision: The number of decimal places for pips (default is 4 for most currency pairs).

                Returns:
                - The number of pips.
                """
                pip_precision=self.calculate_pip_precision(price2)
                smallest_price_movement = 10 ** -pip_precision
                price_difference = price2 - price1
                pips = (price_difference / smallest_price_movement)
                print("Pips:",pips)
                print("pip_precision:",pip_precision)
                return round(pips[0], pip_precision)
        def calculate_pip_precision(self,number):
                """
                Calculate pip precision from a given number.

                Parameters:
                - number: The input number.

                Returns:
                - The number of decimal places (pip precision).
                """
                # Convert the number to a string to handle scientific notation
                number_str = str(number)

                # Check if the number is in scientific notation
                if 'e' in number_str:
                        # Extract the exponent part
                        exponent_part = number_str.split('e')[1]

                        # Calculate the number of decimal places from the exponent
             
                        decimal_places = abs(int(exponent_part))
                else:
                        # Find the position of the decimal point
                        decimal_position = number_str.find('.')

                        # Calculate the number of decimal places
                        decimal_places = len(number_str) - decimal_position - 1

                return decimal_places
        def _next_observation(self):
                observation = np.array(self.currencies.iloc[self.current_step])
                return observation 
        def reset(self,*args,**kwargs):
                super().reset(*args,**kwargs)
                self.account_balance=self.__default_account_balance__
                self.account_equity=self.__default_account_balance__
                self.margin=self.__default_account_balance__
                return np.array([float(self.account_balance),float(self.account_equity),float(self.margin)]),{"active-trades":len(self.open_trades),"outcomes":0,"closed-trades":0}
        def step(self,action):
                """
                A forex trading action can either be a 
                        
                        
                        0.buy -set profit and loss in pips
                        1.sell -set profit and loss in pips
                        2.modify tp /sl (trailing stop loss) 
                        
                of the form 

                option=Dict({"Entry":0.0,LotSize":0.25,"Option":Discrete(2),"TakeProfit":Box(0,70000,(1,1)),"StopLoss":Box(0,70000,(1,1))})

                action ={"pair":"GBPUSD" ,"position":option}
                A sell option is 0 while a buy option is 1 , -1 is modify tp
                """
                print("======Action======\n",action)
                truncated=False
                #self.action=Dict({"LotSize":Box(0,70000,(1,1)),"Option":Discrete(3),"TakeProfit":Box(0,70000,(1,1)),"StopLoss":Box(0,70000,(1,1))})
                
               
                action["Id"]=self.trade_index
                print("Made Action:",action)
                
                done=False
                TakeProfit=action["TakeProfit"]
                StopLoss=action["StopLoss"]
                trades_to_close=[]
                #Actions Evalution
                for trade in self.open_trades:
                        entry_price=trade["Entry"]
                        print("trade pair :",trade["pair"])
                        print(self.currencies[trade["pair"]].iloc[self.current_price])
                        current_high=self.currencies[trade["pair"]].iloc[self.current_price]["High"]
                        current_low=self.currencies[trade["pair"]].iloc[self.current_price]["Low"]
                        print(trade["Entry"])
                        buy_trade_result=self.calculate_pips(current_high,trade["Entry"])
                        sell_trade_result=self.calculate_pips(entry_price,current_low)

                        trade_lot_size=trade["LostSize"]

                        outcome=0 # profit or loss

                        # if trade is a buy
                        if trade["Option"]==0:
                                #Hit buy stop loss
                                if trade["StopLoss"]>=buy_trade_result:
                                        reward=-trade["StopLoss"]
                                        trades_to_close.append(trade)
                                        outcome-=reward*trade_lot_size*self.calculate_pip_precision(entry_price)

                                #Hit sell take profit
                                if trade["TakeProfit"]>=buy_trade_result:
                                        reward+=trade["TakeProfit"]
                                        trades_to_close.append(trade)
                                        outcome+=reward*trade_lot_size*self.calculate_pip_precision(entry_price)
                                
                        #if trade is a sell
                        elif  trade["Option"]==1:
                                if trade["StopLoss"]<=sell_trade_result:
                                        #if trade is in break even mode or in profit mode
                                        if trade["StopLoss"]<=trade["Entry"]:
                                                reward+=trade["StopLoss"]
                                        else:
                                                reward=-trade["StopLoss"]
                                        trades_to_close.append(trade)
                                        outcome-=reward*trade_lot_size*self.calculate_pip_precision(entry_price)
                                if trade["TakeProfit"]<=sell_trade_result:
                                        reward=+trade["TakeProfit"]
                                        trades_to_close.append(trade)
                                        outcome+=reward*trade_lot_size*self.calculate_pip_precision(entry_price)
                                
                        self.account_balance=self.account_balance+outcome
                        if len(self.currencies[trade["Pair"]].shape[0])-1==self.current_price:
                                done=True 
                                
                        #if in profit reward the pips

                        #if in loss remove the pips:
                for i in trades_to_close:
                        self.open_trades.remove(i)   
                #Action Exectution
                if action["Option"]==0 or action["Option"]==1:
                        self.trade_index+=1
                if action["Option"]==1:
                        entry=self.currencies[action["pair"]]["Low"][self.current_price]
                        
                        print(action["TakeProfit"])
                        print(entry)
                        action["Entry"]=entry
                        action["StopLoss"]=np.array([self.calculate_pips(entry,StopLoss )])
                        action["TakeProfit"]=np.array([self.calculate_pips(TakeProfit,entry)])
                        self.open_trades.append(action)
                        print("Made a Sell Trade:",action)
                if action["Option"]==0:
                        entry=self.currencies[action["pair"]]["High"][self.current_price]
                        action["Entry"]=entry
                        print(action["TakeProfit"])
                        print(entry)
                        action["StopLoss"]=np.array([self.calculate_pips(StopLoss ,entry)])
                        action["TakeProfit"]=np.array([self.calculate_pips(TakeProfit,entry)])
                        self.open_trades.append(action)
                        print("Made a Buy Trade:",action)
                if action["Option"]==2:
                        for trade in self.open_trades:
                                if trade["Id"]==action["TradeUid"]:
                                                #if its a buy trade
                                                if trade["Option"]==0:
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss ,trade["Entry"])
                                                        trade["TakeProfit"]=self.calculate_pips(TakeProfit,trade["Entry"])
                                                elif trade["Option"]==1: 
                                                        #trade["StopLoss"]=StopLoss
                                                        trade["TakeProfit"]=self.calculate_pips(trade["Entry"],TakeProfit)
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss,trade["Entry"])

                                        
                #Add price to the environment
               

                
               
                outcome=0
                reward=0
                
               
                state=np.array([float(self.account_balance),float(self.account_equity),float(self.margin)])
                info={"active-trades":len(self.open_trades),"outcomes":outcome,"closed-trades":len(trades_to_close)}
                self.current_price+=1
                return state,reward ,done,truncated,info

env=GBPForexEnvironment(data={"GBPUSD":pd.read_csv(os.path.join("data","GBPUSD_DATA"))})
check_env(env)                    
