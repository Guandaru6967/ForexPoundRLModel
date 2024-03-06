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
from priceprocessor import PriceNormlizer
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical
m=Categorical( torch.tensor([0.2,0.2,.2,2e-1,2.5,0.3]))

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
                activationValue =nn.ReLU()(lstmValue)
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
        def training_step(self, *args: plt.Any, **kwargs: plt.Any):
                super().training_step(*args, **kwargs)
                print("Args:\n",args)
                batch,batchidx=args
                print("batch:\n",batch,"batchidx:\n",batchidx)
                # Unpack batch
                states, actions, rewards, next_states, dones = batch
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
                                                                                       "spread":self.spread_space
                                                                                       })
                self.trade_param_space=gym.spaces.Box(high=8000.00,low=10,shape=(2,))
                print(self.trade_param_space.env)
                self.action_space = gym.spaces.Box(low=np.array([0, 10, 10, 0.01]), high=np.array([2, 1000, 1000, self.lot_size_max]))

                print("sample:",self.action_space.sample())
                print(isinstance(self.action_space,gym.spaces.Tuple))
                
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
                print("Prices 1 and 2:",price1, price2)
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


class GBPTrainer:
        def __init__(self):
                gbpusd=os.path.join("data","GBPUSD_DATA")
                testdata=os.path.join("data","SINE_FXENV_TESTDATA.csv")

                testdata=pd.read_csv(testdata).round(5)
                gbpusd=pd.read_csv(gbpusd).round(5)
                reward_threshold=1000
                env=GBPForexEnvironment(data=testdata,account_balance=100000)
                agent=ForexPPOAgent()
                reward_threshold_callback = RewardThresholdCallback(reward_threshold)

                # Train the agent using a Lightning Trainer with the callback
                trainer = pl.Trainer(callbacks=[reward_threshold_callback])
                trainer.fit(agent, env)



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
        env=GBPForexEnvironment(data=testdata,account_balance=100000)
        #print(env.currencies.iloc[2000])
        

        #for i in range(1000):
        waitaction={"Id":2,"LotSize":np.array([15.00]),"Option":3,"Entry":np.array([1.30000]),"TakeProfit":np.array([1.30500]),"StopLoss":np.array([1.29000])}
        buyaction={"Id":0,"LotSize":np.array([0.15]),"Option":0,"Entry":np.array([1.30000]),"TakeProfit":np.array([1.3200]),"StopLoss":np.array([1.29000])}

        firstaction={"Id":0,"LotSize":np.array([0.15]),"Option":1,"Entry":np.array([0.02]),"TakeProfit":np.array([1.33500]),"StopLoss":np.array([1.3500])}
        sellaction={"Id":0,"LotSize":np.array([0.15]),"Option":1,"Entry":np.array([0.02]),"TakeProfit":np.array([1.31000]),"StopLoss":np.array([1.31000])}
        
        env.step(buyaction)
        print(100)
        for i in range(30):
                env.step(waitaction)
                print(20)
        #env.render()
       
        check_env(env)

def algotest():
        def envProcess():
                env=GBPForexEnvironment(data=testdata,account_balance=100000)
                return env
        num_envs=1
        subprocess_env = DummyVecEnv([envProcess for i in range(num_envs)])
        #model=A2C("MlpPolicy",env=subprocess_env,verbose=1)
        #model.learn(total_timesteps=100000,log_interval=500)
def train():
        trainer=GBPTrainer()
        trainer.train()
        
        
algotest()
        