from talib.abstract import *
from talib import *
import talib
import numpy as np
from talib.stream import *
import pandas as pd
import gymnasium as gym 
from gym_anytrading.envs  import ForexEnv
import contextlib
import yfinance as yf
import datetime
from  stable_baselines3.common.vec_env import *
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback,StopTrainingOnRewardThreshold,StopTrainingOnNoModelImprovement
from sb3_contrib import RecurrentPPO
from stable_baselines3.a2c import A2C
import matplotlib.pyplot as plt
import os
import keras 
from forexfunctions import zigzag
from priceprocessor import *

from sklearn.cluster import KMeans
def readdate(string):
    #datetime.datetime.strftime(string)
    return int(datetime.datetime.fromisoformat(string).timestamp())#.replace("+00:00", "+0000"))

class ForexDataCollection:
    def __init__(self):
        self.pairs=["GBPUSD","GBPAUD","GBPNZD","GBPCAD","GBPCHF","GBPJPY"]
    
        self.pair_data={}
        self.data_startdate="2024-01-01"
        self.data_enddate="2024-02-27"

        for i in self.pairs:
            #self.pair_data[i]=yf.download(i+"=X", start=self.data_startdate, end=self.data_enddate, interval="5m")
            #print(i)
            #print(self.pair_data[i])
            self.pair_data[i]=pd.read_csv(os.path.join("data",i+"_DATA"))
            #self.pair_data[i].index.name = 'DatetimeIndex'
            #help(mpl.plot)
            #mpl.plot(self.pair_data[i],type='candle',title='Candlestick Chart', ylabel='Price')
            #self.pair_data[i]["Datetime"]=self.pair_data[i]["Datetime"].apply(readdate)
            self.pair_data[i].sort_values("Datetime",ascending=True,inplace=True)
            #self.pair_data[i]=self.pair_data[i].drop(columns=["Unnamed: 0","Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1"])
            # self.pair_data[i]=self.pair_data[i].drop(columns=["Unnamed: 0"])
            self.pair_data[i].set_index("Datetime",inplace=True)
            self.pair_data[i].to_csv(os.path.join("data",i+"_DATA"))
   

         
game=pd.DataFrame({"High":[11,45,53,32,24],"Low":[23,60,45,61,12],"Close":[34,67,22,68,52],"Open":[11,13,35,78,36]})

dataCollection=ForexDataCollection()

gbpdata=dataCollection.pair_data["GBPUSD"]

print("zigzag:",gbpdata)


class StopTrainingCallback(BaseCallback):
    def __init__(self, eval_freq, threshold, verbose=0):
        super(StopTrainingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.threshold = threshold

    def _on_step(self) -> bool:
        # Check the explained_variance every `eval_freq` steps
        if self.num_timesteps % self.eval_freq == 0:
            pass
          

        return True 
class TradingRLClass:

    def createEnv(self,x):
        env=gym.make("forex-v0",df=self.training_data,frame_bound=self.frame_bound,window_size=self.window_size)
        env.seed(x)
    def __init__(self,**kwargs):
        self.modelName="gbpSuperAI"
        self.frame_bound=(48,4000)
        self.window_size=48
        pricedata=ForexDataCollection()
        gbpdata=ProcessDataWithAllFunctions(pricedata.pair_data["GBPUSD"])
        self.training_data=gbpdata
        self.testing_data=gbpdata.iloc[5001:]
        self.environment=gym.make("forex-v0",df=self.training_data,frame_bound=self.frame_bound,window_size=self.window_size)
        self.environment_maker=lambda : self.environment
        state=self.environment.reset()
        self.dummyEnvironment=DummyVecEnv([self.environment_maker ] )
    
        #self.model=A2C("MlpPolicy",self.dummyEnvironment,verbose=1)
        self.model=RecurrentPPO("MlpLstmPolicy",self.dummyEnvironment,verbose=1)
        print(self.model.get_parameters())
        print(self.model.policy)
        
        #self.loadModel()
        #quit()
        self.learnCallback= StopTrainingCallback(eval_freq=10000, threshold=0.15)
        self.stopNoImprovementCallback=StopTrainingOnNoModelImprovement(5,2000,1)
        self.evalCallback = EvalCallback(self.dummyEnvironment, eval_freq=1000, callback_after_eval=self.stopNoImprovementCallback, verbose=1)

        self.stopCallback=StopTrainingOnRewardThreshold(300,1)
        self.evalCallback = EvalCallback(self.dummyEnvironment, eval_freq=1000, callback_after_eval=self.stopCallback, verbose=1)

    def buildEnvironment(self):
        while True:
            action=self.environment.action_space.sample()
            n_state, reward, done, truncated, info =self.environment.step(action)
            
            if truncated:
                print("Truncated!")
                break
            if done:
                print("info",info)
                break
    
    def trainRL(self,timesteps=100000):
        self.model.learn(callback=self.evalCallback,total_timesteps=timesteps)
        self.saveModel()
    def loadModel(self):
        try:
            self.model=self.model.load(path=self.modelName,env=self.dummyEnvironment)
            print(f"Model name {self.modelName} found and loaded")
        except FileNotFoundError:
            print(f"Model of name {self.modelName} not found yet")
            pass
    def saveModel(self):
        self.model.save(self.modelName)
    def visualize(self):
        plt.figure(figsize=(15,6))
        plt.cla()
        self.environment.render()
        plt.show()
    def testData(self):
        env=gym.make("forex-v0",df=self.testing_data,frame_bound=self.frame_bound,window_size=self.window_size)
        
        obs=env.reset()
        print(obs[0],obs[0].shape)
        #quit()
        for index,row in self.testing_data.iterrows():
            action,states=self.model.predict(row,deterministic=True)
            print(action,states)
            n_state,reward,done,truncated,info=env.step(action)
            print("Observavtion:",reward,info)
            if done:
                print("Last Observavtion :",info)
                break
        while False:
            action,states=self.model.predict(obs[0],deterministic=True)
            n_state,reward,done,truncated,info=env.step(action)
            print("Observavtion:",reward)
            if done:
                print("Last Observavtion :",info)
                break

trading=TradingRLClass()
trading.trainRL()


#trading.testData()
#trading.testData(dataCollection.pair_data["GBPAUD"])
#trading.visualize()






