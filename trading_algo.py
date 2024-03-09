from matplotlib.pylab import Generator
import matplotlib.pyplot as plt
import matplotlib
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

from torch.utils.data.dataset import ConcatDataset
from priceprocessor import PriceNormlizer
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader,Dataset
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
class ForexDataLoader(DataLoader):
        pass
class ForexEnvDataSet(Dataset):
    def __init__(self, batchdata:list):
        self.batchdata = batchdata

    def __len__(self):
        return len(self.batchdata)
    def __call__(self,value):
            return self.batchdata.append(value)
    def __getitem__(self, idx):
        return self.batchdata[idx]
class ObservationSpace(gym.Space):
        def __init__(self, *args,**kwargs):
                super().__init__(*args,**kwargs)
                print(kwargs)
                print(args)
class RewardThresholdCallback(pl.Callback):
    def __init__(self, reward_threshold):
        super().__init__()
        self.reward_threshold = reward_threshold

    def on_epoch_end(self, trainer, pl_module):
        if pl_module.current_reward >= self.reward_threshold:
            print(f"Training stopped as reward threshold ({self.reward_threshold}) reached.")
            trainer.should_stop = True
class ForexPolicyLSTMNeuralNetwork(nn.Module):
        def __init__(self,input_size,action_size,*args,hidden_size=912,**kwargs):
                super().__init__(*args,**kwargs)
                self.lstmLayer=nn.LSTM(input_size=input_size,hidden_size=hidden_size)
                
                self.fullyconnectedLayer=nn.Linear(hidden_size,action_size)

        def forward(self, x):
                # Forward pass through LSTM layer
                lstmValue, _ = self.lstmLayer(x)
                # Apply ReLU activation function
                activationValue =F.relu(lstmValue)
                # Forward pass through fully connected layer
                y = self.fullyconnectedLayer(activationValue)
                return y
class ForexPPOAgent(pl.LightningModule):
        def __init__(self,input_size,action_size,gamma=99e-2,learning_rate=1e-3,clip_ratio=2e-1,value_coefficient=5e-1,entropy_coefficient=1e-2):
                super().__init__()
                
                self.policy=ForexPolicyLSTMNeuralNetwork(input_size=input_size,action_size=action_size)
                self.gamma=gamma
                self.learning_rate=learning_rate
                self.clip_ratio=clip_ratio
                self.value_coefficient=value_coefficient
                self.entropy_coefficient=entropy_coefficient
        def configure_optimizers(self):
                super().configure_optimizers()
                optimizer=torch.optim.Adam(params=self.policy.parameters(),lr=self.learning_rate)
                print("policy params to be optimized:",self.policy.parameters())
                return optimizer
        def training_step(self, *args, **kwargs):
                super().training_step(*args, **kwargs)
                print("Args:\n",args)
                batch,batchidx=args
                print("batch:\n",batch,"batchidx:\n",batchidx)
                # Unpack batch
                states, actions, rewards, next_states, dones = batch
                print("states:",states)
                print("netx_states:",next_states)
                # Forward pass through policy network
                logits = self.policy(states)
                # Initialize Categorical distribution
                dist = Categorical(logits=logits)
                # Calculate log probabilities of actions
                log_probs = dist.log_prob(actions)
                # Forward pass through policy network for value estimation
                values = self.policy(states)
                # Forward pass through policy network for value estimation for next states
                next_values = self.policy(next_states)
                # Compute advantages and returns
                returns, advantages = self._compute_advantages(rewards, values, next_values, dones)

                # Compute surrogate objective function for policy loss
                ratios = torch.exp(log_probs - log_probs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value function loss
                value_loss = F.smooth_l1_loss(values, returns.detach())

                # Compute entropy bonus
                entropy = dist.entropy().mean()

                # Compute total loss
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                return loss
        def validation_step(self, batch, batch_idx):
                states, actions, rewards, next_states, dones = batch
                logits = self.policy(states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                values = self.policy(states)
                next_values = self.policy(next_states)
                returns, advantages = self._compute_advantages(rewards, values, next_values, dones)

                # Policy loss
                ratios = torch.exp(log_probs - log_probs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = F.smooth_l1_loss(values, returns.detach())

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                return loss
        def validation_epoch_end(self, outputs):
                avg_loss = torch.stack(outputs).mean()
                self.log('val_loss', avg_loss)
        def collectdata(self):
                pass

class PPOAgent(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_actions, lr=1e-3, gamma=0.99, clip_ratio=0.2, vf_coef=0.5, entropy_coef=0.01):
        super(PPOAgent, self).__init__()
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
                
                self.currencies=data#PriceNormlizer(data).round(6)
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
                self.trade_outcomes=[]
                self.lot_size_max=(risk_size*self.account_balance)/(20*self.calculate_pip_precision(self.max_price))
                print("Max Lot size: ", self.lot_size_max)
                self.temporal_window=temporal_window
                self.lot_ratio=0
                self.risk_reward_ratio=2.0
                self.open_trades=[]
                self.currency_spread=currency_spread
                self.max_trades=10
                self.trade_index=0
                self.average_profit_pips=3000
                self.average_loss_pips=2000
                self.track_data={"profits":[],"balance":[],"losses":[],"rewards":[],"steps":[],"trades":[]}
        
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
        
                """
                self.action_space=
                {
                "Id" : gym.spaces.Discrete(self.max_trades),
                "LotSize" : self.lot_size_space,
                "Option" : self.trade_option_space,
                "Entry" : self.trade_entry_space,
                "TakeProfit" : self.take_profit_space,
                "StopLoss" : self.stop_loss_space
                }
                """
                
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
                self.trades_space=gym.spaces.Sequence(self.trade_entry_space)#(self.action_space)
                
                self.observation_space=gym.spaces.Dict({"temporal_window_state":gym.spaces.Box(low=self.min_price,
                                                                                       high=self.max_price,
                                                                                       shape=(self.temporal_window,(len(data.columns.to_list())-1)),dtype=np.float64) ,
                                                                                       "balance":self.balance_space,
                                                                                       "equity":self.equity_space,
                                                                                       "margin":self.margin_space,
                                                                                       "spread":self.spread_space,
                                                                                       "profit":gym.spaces.Box(high=10e1000,low=0,dtype=np.float64),
                                                                                       "loss":gym.spaces.Box(high=10e1000,low=0,dtype=np.float64)
                                                                                       })
                self.trade_param_space=gym.spaces.Box(high=8000.00,low=10,shape=(2,))
                print(self.trade_param_space)
                #[Option,TakeProfit,StopLoss,lot_size,modification target]
                self.action_space = gym.spaces.Box(low=np.array([0.0, 10, 10, 0.01,0]), high=np.array([3.00, 1000, 1000, self.lot_size_max,self.max_trades]))

                print("sample:",self.action_space.sample())
                print("sample shape:",self.action_space.shape)
                #print(isinstance(self.action_space,gym.spaces.Tuple))
                
                self.obs_sample=self.observation_space.sample()
                print("Observation:",self.obs_sample)
        def calculate_pips(self,price1, price2,precision=5):
                """
                Calculate the number of pips between two prices.

                Parameters:
                - price1: The first price.
                - price2: The second price.
                - pip_precision: The number of decimal places for pips (default is 4 for most currency pairs).

                Returns:
                - The number of pips.
                """
                pip_precision=precision
                smallest_price_movement = 10 ** -pip_precision
                price_difference = price2 - price1
                pips = (price_difference / smallest_price_movement)
                # print("Prices 1 and 2:",price1, price2)
                # print("Pips:",pips)
                # print("pip_precision:",pip_precision)
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
                wait_option={'Entry':self.currencies.iloc[self.current_price]["High"], 'Id': self.trade_index, 'LotSize': 0.0, 'Option': 3, 'StopLoss': self.max_price, 'TakeProfit':self.max_price}
               
                #return self.obs_sample,{"active-trades":len(self.open_trades),"outcomes":0,"closed-trades":0}
                reset_obs={"temporal_window_state":self.currencies.iloc[self.current_price:self.current_price+self.temporal_window][self.features_fields].to_numpy(),
                                                                                       "balance":np.array([float(self.account_balance)]),
                                                                                       "equity":np.array([float(self.account_equity)]),
                                                                                       "margin":np.array([float(self.account_balance)]),
                                                                                       "spread":np.array([float(self.currency_spread)]),
                                                                                        "profit":np.array([float(self.total_profits)]),
                                                                                        "loss":np.array([float(self.total_losses)])
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
                
               
                print("Made Action:",action)
                
                done=False
                
                trades_to_close=[]
                outcome=0
                reward=0
                
                #Trades evalution with respect to current trade
                for trade in self.open_trades:
                        
                        current_high=self.currencies.iloc[self.current_price]["High"]
                        current_low=self.currencies.iloc[self.current_price]["Low"]
                      
                       

                       

                        outcome=0 # profit or loss
                        trade_entry=trade["Entry"]
                        trade_stop_loss=trade['StopLossPips']
                        trade_take_profit=trade["TakeProfitPips"]
                        trade_lot_size=trade["LotSize"]

                        trade_id=trade["Id"]
                        trade_option=trade["Option"]
                        #Current  price  relative to the trade entrys 
                        buy_trade_result=self.calculate_pips(trade_entry,current_high)
                        sell_trade_result=self.calculate_pips(current_low,trade_entry)
                
                        buy_trade_stop=self.calculate_pips(current_low,trade_entry)
                        sell_trade_stop=self.calculate_pips(trade_entry,current_high)
                        print("Trade Entry:",trade_entry,current_high)
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
                        if type(outcome) is not int and type(outcome) is not float and type(outcome) is not np.float64 :
                                print(type(outcome))
                                outcome=outcome[0]
                        
                        #Update account balance
                        self.account_balance=self.account_balance+outcome
                        #Keep record of trade
                        self.trade_outcomes.append({"profit":outcome if outcome>0 else 0,"loss":abs(outcome) if outcome<0 else 0,"entry":trade_entry,"option":"buy" if trade_option==0 else "sell" })    
                        #if in profit reward the pips

                        #if in loss remove the pips:
                print(len(trades_to_close),trades_to_close,self.open_trades)
                for i in trades_to_close:
                        self.open_trades.remove(i)   
                #Action Exectution
                TakeProfit=action[1]
                StopLoss=action[2]
                Option=int(round(action[0],0))
                
                LotSize=action[3]
                ModificationId=action[4]
              
                currentprice=self.currencies.iloc[self.current_price]["High"]
                #Sell Action
                if Option==1:
                        print("Made a Sell Trade:" ,action)
                #Buy Action
                if Option==0:
                        print("Made a Buy Trade",action)
                #Modification Option
                if Option==2:
                        print("Made a Modification Action to trade:",action)
                        for trade in self.open_trades:
                                if trade["Id"]==int(round(ModificationId,0)):
                                                print(f"Modification made to trade {trade['Id']}:",trade)
                                                #if its a buy trade
                                                trade["StopLoss"]=StopLoss
                                                trade["TakeProfit"]=TakeProfit
                                                print("New trade:",trade)
                #Wait option
                if Option==3:
                        reward+=1
                        pass
                if Option==1 or Option==0 :
                        reward==0
                        
                #Ajust environment based on buy or sell trades
                if Option==1 or Option==0 and  len(self.open_trades)<self.max_trades:
                        self.track_data["trades"].append(action)
                        #reward is dedacted if trading ruls for avergae profit and loss are broken
                        reward-=(self.average_loss_pips-StopLoss)
                        reward-=(self.average_profit_pips-TakeProfit)
                        #dd 
                        trade={"Id":self.get_nextid(),"Entry":currentprice,"StopLossPips":StopLoss,"TakeProfitPips":TakeProfit,"Option":Option,"LotSize":LotSize}
                        self.open_trades.append(trade)
                else :
                        #Punish for more trying to execute more trades
                        reward-=10
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
                                                                                       "profit":np.array([float(self.total_profits)]),
                                                                                        "loss":np.array([float(self.total_losses)])
                                                                                       }
                info={"active-trades":len(self.open_trades),"outcomes":outcome,"closed-trades":len(trades_to_close)}
                self.current_price+=1
                return state,reward ,done,truncated,info
        def get_nextid(self):
                ids=[ i["Id"] for i  in self.open_trades]
                labels=[i for i in range(3)]
                for i in labels:
                        if i not in ids:
                                return i
                return np.random.randn()
        def render(self)->None:
                fig, axs = plt.subplots(4)
                x=self.track_data["steps"]
                y1=self.track_data["profits"]
                y2=self.track_data["losses"]
                y3=self.track_data["rewards"]
                y4=self.track_data["balance"]
                # Plot each array in a separate subplot
                print(x,":",y4)
                axs[0].plot(x, y1, color='red', marker='o', linestyle='-')
                axs[0].set_title('Profits')

                axs[1].plot(x, y2, color='blue', marker='x', linestyle='--')
                axs[1].set_title('Losses')

                axs[2].plot(x, y3, color='green', marker='s', linestyle='-.')
                axs[2].set_title('Rewards')

                axs[3].plot(x, y4, color='purple', marker='x', linestyle='-')
                axs[3].set_title('Account Balance')
                plt.show()
        @property
        def total_profits(self):
                return sum([i["profit"] for i in self.trade_outcomes])
        @property
        def total_losses(self):
                return sum([i["loss"] for i in self.trade_outcomes])

class GBPTrainer:
        def __init__(self,traindata:pd.DataFrame):
                gbpusd=os.path.join("data","GBPUSD_DATA")
                testdata=os.path.join("data","SINE_FXENV_TESTDATA.csv")

                testdata=pd.read_csv(testdata).round(5)
                gbpusd=pd.read_csv(gbpusd).round(5)
                reward_threshold=1000
                self.environment=GBPForexEnvironment(data=traindata,account_balance=100000)
            
                self.agent=ForexPPOAgent(input_size=len(traindata.columns.to_list()),action_size=self.environment.action_space.shape[0])
                reward_threshold_callback = RewardThresholdCallback(reward_threshold)

                # Train the agent using a Lightning Trainer with the callback
                self.trainer = pl.Trainer(max_epochs=2000,callbacks=[reward_threshold_callback])
                
        def train(self):
                self.agent=self.agent.train()
                self.dataloader=DataLoader(dataset=self.collectEnvdata())
                self.trainer.fit(self.agent,train_dataloaders=self.dataloader) 
                
        def collectEnvdata(self):
                
                
                dataset=ForexEnvDataSet([])
                for epoch in range(self.trainer.max_epochs):
                        state = self.environment.reset()  # Reset the environment
                        done = False
                        while not done:
                                # Interact with the environment
                                logits = self.agent.policy(state)  # Forward pass through policy network to get logits
                                action_probs = F.softmax(logits, dim=-1)  # Convert logits to action probabilities
                                action = torch.multinomial(action_probs, num_samples=1) 
                                print("Action taken by policy:",action)
                                next_state, reward, done, _ = self.environment.step(action)  # Take action in the environment
                                
                                # Collect data for training
                                dataset(torch.tensor(data=[state, action, reward, next_state, done]))
                                
                                state = next_state  # Update the current state
                                
                return dataset



def make_env( rank, seed=0):
    def _init():
        env = GBPForexEnvironment(data=pd.read_csv(os.path.join("data","GBPUSD_DATA")))
          # Set a unique seed for each environment
        return env
    return _init

# Create vectorized environment with seed

print(120)
gbpusd=os.path.join("data","GBPUSD_DATA")
testdata=os.path.join("data","SINE_FXENV_TESTDATA.csv")

testdata=pd.read_csv(testdata).round(5)
gbpusd=pd.read_csv(gbpusd).round(5)
def test():
        env=GBPForexEnvironment(data=testdata[1000:],account_balance=100000)
        #print(env.currencies.iloc[2000])
        #print(env.action_space.sample())

        #for i in range(1000):
        waitaction=np.array( [3.00, 3020, 30500, 15.00, 00000])
        buyaction= np.array([0.00, 3020, 2900, 20.00, 00000])

        firstaction= np.array([0.00, 1.33500, 1.3500, 15.00, 00000])
        
        sellaction =np.array([1.00, 2120, 1000, 15.00, 00000])
        
        env.step(sellaction)
        print(100)
        for i in range(30):
                env.step(waitaction)
                print(20)
        #env.render()
       
        #check_env(env)

def algotest(data:pd.DataFrame):
        trainer=GBPTrainer(testdata)
        trainer.train()
def train():
        trainer=GBPTrainer()
        trainer.train()
        
#test()       
algotest(testdata)
        