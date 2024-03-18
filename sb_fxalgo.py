from sb3_contrib.ppo_recurrent import RecurrentPPO,MultiInputLstmPolicy
from fxenvironment import GBPForexEnvironment
from stable_baselines3.common.utils import check_for_correct_spaces
from priceprocessor import ProcessDataWithAllFunctions,PirceDataNormalizer
import os 
import pandas as pd
from torchinfo import summary
from stable_baselines3.common.env_checker import check_env

def datadynamicprocess(df):
        import pandas as pd

        dataframe=pd.DataFrame()
        
        dataframe[[ i.title().replace("<","").replace(">","")  for i in df.columns.to_list()]]=df[[i for i in df.columns.to_list()]]
        
        

        dataframe["Date"]=pd.to_datetime(dataframe["Date"],format='%Y.%m.%d')

        dataframe.set_index("Date",inplace=True)
        dataframe=dataframe.drop("Vol",axis=1)
        
        dataframe=dataframe.dropna()
        

        return dataframe
from threading import Thread
from multiprocessing import Process
def ModelTrain():
        import mplfinance as mplf
        datapath=os.path.join("data/GBPUSD5MIN","GBPUSD_M5_2020_01_06_0000_2023_09_04_0045.csv")
        datasize=40000
        dataframe=datadynamicprocess(pd.read_csv(datapath)[:datasize])
        mplf.plot(dataframe, type='candle', style='charles', title='OHLC Chart', ylabel='Price')
        mplf.show()
        print("Running...")
        dataframe=PirceDataNormalizer(ProcessDataWithAllFunctions(dataframe))


        environment=GBPForexEnvironment(dataframe,account_balance=100_000)
        #check_env(environment)

        
        #quit()
        ppo_model=RecurrentPPO("MultiInputLstmPolicy",ent_coef=0.05,learning_rate=0.05,env=environment,policy_kwargs={"n_lstm_layers":4,"enable_critic_lstm":True})
        
        try:
                ppo_model=ppo_model.load("fx_ppo",env=environment)
                print("Loaded model")
        except: 
                print("Failed to load model")
                pass
        print("Learning....")
        
        ppo_model.learn(total_timesteps=25000)
        
        ppo_model.save("fx_ppo")
        environment.render()
        quit()
        #Evaluation 
        test_dataframe=PirceDataNormalizer(ProcessDataWithAllFunctions(datadynamicprocess(pd.read_csv(datapath)[20000:40000])))
        test_environment=GBPForexEnvironment(test_dataframe,account_balance=100_000)
        ppomodel=RecurrentPPO("MultiInputLstmPolicy",env=test_environment,policy_kwargs={"n_lstm_layers":4,"enable_critic_lstm":True},verbose=1)
        ppomodel=ppo_model.load("fx_ppo",env=environment)
        obs=test_environment.reset()
        count=0
        while True:
                action=ppomodel.predict(obs)
                obs,reward ,done,truncated,info=test_environment.step(action)
                if count==10000:
                        break
                count+=1
        test_environment.render()

                
        
        # for module in  ppo_policy.policy.modules():
        #         print(summary(module))

def fxtest():
        from gym_anytrading.envs import ForexEnv
        datapath=os.path.join("data/GBPUSD5MIN","GBPUSD_M5_2020_01_06_0000_2023_09_04_0045.csv")
        dataframe=datadynamicprocess(pd.read_csv(datapath)[:20000])
        dataframe=PirceDataNormalizer(ProcessDataWithAllFunctions(dataframe))
        environment=ForexEnv(dataframe,100,frame_bound=(100,dataframe.shape[0]))
        ppo_model=RecurrentPPO("MlpLstmPolicy",env=environment,policy_kwargs={"n_lstm_layers":4,"shared_lstm":True})
        
        print("Learning")
        ppo_model=ppo_model.load("fx_ppo")
        ppo_model.learn(total_timesteps=1000)
        ppo_model.save("fx_ppo")

#fxtest()
ModelTrain()

