from matplotlib.pylab import Generator
from  stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv,VecEnv,SubprocVecEnv

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
        def __init__(self,data:dict,*args,temporal_window=912,account_balance=100.00,risk_size=0.3,currency_spread=0.0,**kwargs) -> None:
                """
                data: 
                {
                        "GBPUSD": pd.DataFrame()
                
                }
                temporal_window: how much of the past data to learn from , defaults to 912(5min data [19, 4Hr candles])
                account_balance:account balance 
                risk_size: what percentage to risk at a time considering maximum amount of trades
                currency_spread: spread of the currency

                """
                super().__init__(*args,**kwargs)
                self.currencies=data
                self.max_price=0.0
                self.min_price=0.00
                count=0
                
                self.current_price=0
                
                self.currencies=PriceNormlizer(data)
                self.features_fields=[i for i in self.currencies.columns.to_list() if i!="Datetime"]
                self.max_price=max([self.max_price,self.currencies["High"].max()])
                self.min_price=min([self.min_price, self.currencies["Low"].min()])
                print(self.max_price,self.min_price)
                
                count+=1
                
                #Account Management settings
                self.account_balance=account_balance
                self.__default_account_balance__=self.account_balance
                self.account_equity=self.__default_account_balance__
                self.margin=self.account_balance
                self.lot_size_max=(risk_size*self.account_balance)/(20*self.calculate_pip_precision(self.max_price))
                print("Max Lot size: ", self.lot_size_max)
                self.temporal_window=temporal_window
                self.lot_ratio=0
                self.risk_reward_ratio=2.0
                self.open_trades=[]
                self.currency_spread=currency_spread
                self.max_trades=10
                self.trade_index=0
                self.average_profit_pips=30
                self.average_loss_pips=20
                #Trade Execution Spaces
                self.lot_size_space=gym.spaces.Box(low=0.01,high=self.lot_size_max)
                self.stop_loss_space=gym.spaces.Box(low=float(self.min_price),high=float(self.max_price))
                self.trade_option_space=gym.spaces.Discrete(4)
                self.trade_entry_space=gym.spaces.Box(low=float(self.min_price),high=float(self.max_price))
                self.take_profit_space=gym.spaces.Box(high=float(self.max_price),low=float(self.max_price))
                # Account Spaces
                self.balance_space=gym.spaces.Box(high=float(100000000.00),low=float(0.0),dtype=np.float64)
                self.equity_space=gym.spaces.Box(high=float(100000000.00),low=float(0.0),dtype=np.float64)
                self.margin_space=gym.spaces.Box(high=float(100000000.00),low=float(0.0),dtype=np.float64)
                self.spread_space=gym.spaces.Box(high=float(0.00001),low=float(0.0),dtype=np.float64)
        
                self.action_space=gym.spaces.Dict({"Id":gym.spaces.Discrete(self.max_trades),"LotSize":self.lot_size_space,"Option":self.trade_option_space,"Entry":self.trade_entry_space,"TakeProfit":self.take_profit_space,"StopLoss":self.stop_loss_space})
                print("sample:",self.action_space.sample())
                #self.observation_space =gym.spaces.Dict(low=0.0,high=1000000.0,shape=(3,),dtype=np.float64)#Dict({"balance":Box(0,100000000,shape=(1,)),"equity":Box(0,100000000,shape=(1,)),"margin":Box(0,100000000,shape=(1,))})  
                self.trade_npdtype= np.dtype([
    ('Entry', float),       # 'Entry' field with float data type
    ('Id', int),            # 'Id' field with int data type
    ('LotSize', float),     # 'LotSize' field with float data type
    ('Option', int),        # 'Option' field with int data type
    ('StopLoss', float),    # 'StopLoss' field with float data type
    ('TakeProfit', float)   # 'TakeProfit' field with float data type
                ])
                print(self.trade_entry_space.low)
                
                low = np.array([self.trade_entry_space.low[0], 0,self.lot_size_space.low[0], 0, self.stop_loss_space.low[0],self.take_profit_space.low[0]])
                high = np.array([self.trade_entry_space.high[0], 1000000, self.lot_size_space.high[0], 4, self.stop_loss_space.high[0],self.take_profit_space.high[0]])
                self.observation_space=gym.spaces.Dict({"temporal_window_state":gym.spaces.Box(low=self.min_price,
                                                                                       high=self.max_price,
                                                                                       shape=(self.temporal_window,(len(data.columns.to_list())-1)),dtype=np.float64) ,
                                                                                       "balance":self.balance_space,
                                                                                       "equity":self.equity_space,
                                                                                       "margin":self.margin_space,
                                                                                       "spread":self.spread_space,
                                                                                       
                                                                                       })
                print(isinstance(self.action_space,gym.spaces.Dict))
                

                self.obs_sample=self.observation_space.sample()
                print("Observation:",self.obs_sample)
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
                return round(pips, pip_precision)
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
                self.open_trades=[]
                print(self.currencies.iloc[self.current_price:self.current_price+self.temporal_window][self.features_fields].to_numpy().shape)
                self.trade_index=0
                self.current_price=0
                np.dtype
                wait_option={'Entry':self.currencies.iloc[self.current_price]["High"], 'Id': self.trade_index, 'LotSize': 0.0, 'Option': 3, 'StopLoss': self.max_price, 'TakeProfit':self.max_price}
               
                #return self.obs_sample,{"active-trades":len(self.open_trades),"outcomes":0,"closed-trades":0}
                reset_obs={"temporal_window_state":self.currencies.iloc[self.current_price:self.current_price+self.temporal_window][self.features_fields].to_numpy(),
                                                                                       "balance":np.array([float(self.account_balance)]),
                                                                                       "equity":np.array([float(self.account_equity)]),
                                                                                       "margin":np.array([float(self.account_balance)]),
                                                                                       "spread":np.array([float(self.currency_spread)]),
                                                                                    
                                                                                       }
                
                
                print("reset data: ",reset_obs)
                return reset_obs,{"active-trades":len(self.open_trades),"outcomes":0,"closed-trades":0,"open-trades":self.open_trades}
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
                
               
                action["Id"]=self.trade_index
                print("Made Action:",action)
                
                done=False
                TakeProfit=action["TakeProfit"][0]
                StopLoss=action["StopLoss"][0]
                trades_to_close=[]
                
                #Actions Evalution
                for trade in self.open_trades:
                        entry_price=trade["Entry"]
                        print("trade:",trade)
                        current_high=self.currencies.iloc[self.current_price]["High"]
                        current_low=self.currencies.iloc[self.current_price]["Low"]
                        print("Trade Entry:",trade["Entry"],current_high)
                        buy_trade_result=self.calculate_pips(current_high,trade["Entry"])
                        sell_trade_result=self.calculate_pips(entry_price,current_low)

                        trade_lot_size=trade["LotSize"]

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
                        
                                
                        #if in profit reward the pips

                        #if in loss remove the pips:
                for i in trades_to_close:
                        self.open_trades.remove(i)   
                #Action Exectution
                if action["Option"]==0 or action["Option"]==1:
                        self.trade_index+=1
                if action["Option"]==1:
                        entry=self.currencies["Low"][self.current_price]
                        print(action["TakeProfit"])
                        print(entry)
                        action["Entry"]=entry
                        action["StopLoss"]=np.array([self.calculate_pips(entry,StopLoss )])
                        action["TakeProfit"]=np.array([self.calculate_pips(TakeProfit,entry)])
                        self.open_trades.append(action)
                        print("Made a Sell Trade:",action)
                if action["Option"]==0:
                        entry=self.currencies["High"][self.current_price]
                        action["Entry"]=entry
                        print(action["TakeProfit"])
                        print(entry)
                        action["StopLoss"]=np.array([self.calculate_pips(StopLoss ,entry)])
                        action["TakeProfit"]=np.array([self.calculate_pips(TakeProfit,entry)])
                        self.open_trades.append(action)
                        print("Made a Buy Trade:",action)
                if action["Option"]==2:
                        for trade in self.open_trades:
                                if trade["Id"]==action["Id"]:
                                                #if its a buy trade
                                                if trade["Option"]==0:
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss ,trade["Entry"])
                                                        trade["TakeProfit"]=self.calculate_pips(TakeProfit,trade["Entry"])
                                                elif trade["Option"]==1: 
                                                        #trade["StopLoss"]=StopLoss
                                                        trade["TakeProfit"]=self.calculate_pips(trade["Entry"],TakeProfit)
                                                        trade["StopLoss"]=self.calculate_pips(StopLoss,trade["Entry"])
                if action["Option"]==3:
                        pass
                                        
                #Add price to the environment
               

                
                if self.currencies.shape[0]-1==self.current_price:
                                done=True 
                if self.currencies.shape[0]-1==self.current_price-self.temporal_window:
                        done=True
                outcome=0
                reward=0
                
                
                #state=np.array([float(self.account_balance),float(self.account_equity),float(self.margin)])
                state={"temporal_window_state":self.currencies.iloc[self.current_price:self.current_price+self.temporal_window][self.features_fields].to_numpy(),
                                                                                       "balance":np.array([float(self.account_balance)]),
                                                                                       "equity":np.array([float(self.account_equity)]),
                                                                                       "margin":np.array([float(self.account_balance)]),
                                                                                       "spread":np.array([float(self.currency_spread)]),
                                                                                     
                                                                                       }
                info={"active-trades":len(self.open_trades),"outcomes":outcome,"closed-trades":len(trades_to_close)}
                self.current_price+=1
                return state,reward ,done,truncated,info

def make_env( rank, seed=0):
    def _init():
        env = GBPForexEnvironment(data=pd.read_csv(os.path.join("data","GBPUSD_DATA")))
          # Set a unique seed for each environment
        return env
    return _init

# Create vectorized environment with seed
num_envs = 4
env_ids = ['CartPole-v1'] * num_envs
vec_env = DummyVecEnv([make_env(i, seed=42) for i  in enumerate(env_ids)])

quit()
env=GBPForexEnvironment(data=pd.read_csv(os.path.join("data","GBPUSD_DATA")))
environ_maker=lambda x:env
env.reset()
env.step(env.action_space.sample())
env.step(env.action_space.sample())
#env.step()
check_env(env)                    
