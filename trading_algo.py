from matplotlib.pylab import Generator
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
# from keras.layers import Activation,MaxPool3D,AvgPool3D,ConvLSTM3D,LSTM,Dense
# from keras.activations import linear,relu,softmax,sigmoid
# from keras.optimizers import Adam,Adamax 
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from gym.spaces import Box,MultiDiscrete,MultiBinary,Dict,Discrete
import gymnasium as gym
import pandas as pd
import os
import tqdm
from collections import defaultdict


from torch.utils.data.dataset import ConcatDataset
from torchrl.envs.utils import RandomPolicy
from priceprocessor import PriceNormlizer
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader,Dataset

from torchrl.collectors.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from tensordict.nn.distributions import NormalParamExtractor

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch.optim import Adam
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.envs import (TransformedEnv,Compose,DoubleToFloat,StepCounter,ObservationNorm)
from torchrl.objectives.value import GAE
from stable_baselines3 import PPO
from torchinfo import summary

from fxenvironment import FXTorchEnv 
from convoluted_fx import ForexActorNeuralNetwork,ForexCriticNeuralNetwork
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from gym_anytrading.envs import ForexEnv
from sb3_contrib import RecurrentPPO
from priceprocessor import ProcessDataWithAllFunctions,PirceDataNormalizer
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
 

class ForexModelTrainEval():
        def datadynamicprocess(self,df):

                dataframe=pd.DataFrame()
                
                dataframe[[ i.title().replace("<","").replace(">","")  for i in df.columns.to_list()]]=df[[i for i in df.columns.to_list()]]
                
                

                dataframe["Date"]=pd.to_datetime(dataframe["Date"],format='%Y.%m.%d')

                dataframe.set_index("Date",inplace=True)
                dataframe=dataframe.drop("Vol",axis=1)
                
                dataframe=dataframe.dropna()
                

                return dataframe

        def __init__(self,dataframe):
                #Data preprocessing
                self.dataframe=ProcessDataWithAllFunctions(self.datadynamicprocess(dataframe))
                print(self.dataframe.head(1))
                
                self.TEMPORAL_WINDOW=912
                
                #Environment processing
                self.base_env=FXTorchEnv(self.dataframe,temporal_window=self.TEMPORAL_WINDOW)
                #check_env_specs(self.base_env)
                self.environment = TransformedEnv(
                self.base_env,
                Compose(
                      StepCounter(),
                        DoubleToFloat(),
                        
                       #ObservationNorm(in_keys=["temporal_window_state"])
                )
                )
                #self.environment.transform[2].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
                check_env_specs(self.environment)
                #Hyperparamters
                self.sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
                self.batch_size=self.sub_batch_size*10
                self.num_epochs = 10  # optimisation steps per batch of data collected
                self.clip_epsilon = (
                0.2  # clip value for PPO loss: see the equation in the intro for more context.
                )
                self.gamma = 0.99
                self.lmbda = 0.95
                self.entropy_eps = 1e-4
                self.learning_rate=3e-4
                self.FRAMES_PER_BATCH=512#self.TEMPORAL_WINDOW
                self.TOTAL_FRAMES=self.FRAMES_PER_BATCH*self.batch_size

                #Actor(Policy) and Critic (Value) NNs
                self.policynetwork=ForexActorNeuralNetwork(len(self.dataframe.columns.to_list()),self.TEMPORAL_WINDOW)
                
                self.criticnetwork=ForexCriticNeuralNetwork(len(self.dataframe.columns.to_list()),self.TEMPORAL_WINDOW)
                
                self.policy_tensordict=TensorDictModule(module=self.policynetwork,in_keys=["temporal_window_state"],out_keys=["loc", "scale"])
                
                print("rnd",self.environment.action_spec.rand())
                #Policy and Value Modules
                self.policy_module = ProbabilisticActor(
                        module=self.policy_tensordict,
                        spec=self.environment.action_spec,
                        in_keys=["loc", "scale"],
                        distribution_class=TanhNormal,
                        distribution_kwargs={
                                "min": self.environment.action_spec.space.low,
                                "max": self.environment.action_spec.space.high,
                        },
                        return_log_prob=True,
                        # we'll need the log-prob for the numerator of the importance weights
                        )
                #print(self.policy_module(TensorDictModule({"observation":self.environment.reset()})))
                #quit()
                self.value_module = ValueOperator(
                        module=self.criticnetwork,
                        in_keys=["temporal_window_state"],
                        )
                
               
                #Collector Module
                random_policy=RandomPolicy(self.environment.action_spec)
                self.collector = SyncDataCollector(
                        self.environment,
                        self.policy_module,
                        frames_per_batch=self.FRAMES_PER_BATCH,
                        total_frames=self.TOTAL_FRAMES,split_trajs=False
                        )
                #Replay Buffer module
                self.replay_buffer = ReplayBuffer(
                        storage=LazyTensorStorage(max_size=self.FRAMES_PER_BATCH),
                        sampler=SamplerWithoutReplacement(),
                        )
                #General Advantage Estimation Module
                self.advantage_module=GAE(value_network=self.value_module,gamma=self.gamma,lmbda=self.lmbda,average_gae=True,)
                #Loss Function Module
                self.loss_module = ClipPPOLoss(
                actor_network=self.policy_module,
                critic_network=self.value_module,
                clip_epsilon=self.clip_epsilon,
                entropy_bonus=bool(self.entropy_eps),
                entropy_coef=self.entropy_eps,
                # these keys match by default but we set this for completeness
                critic_coef=1.0,
                loss_critic_type="smooth_l1",
                )

                self.optimizer = torch.optim.Adam(self.loss_module.parameters(), self.learning_rate)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.TOTAL_FRAMES // self.FRAMES_PER_BATCH, 0.0
                )
                print("All modules to train the model have been initialized.")
        def train(self):
                logs = defaultdict(list)
                pbar = tqdm.tqdm(total=self.TOTAL_FRAMES)
                eval_str = ""
                print("Loading data")
                for i, tensordict_data in enumerate(self.collector):
                        # we now have a batch of data to work with. Let's learn something from it.
                        print(f"\ntraining data indexed at {i} from the collecton\n")
                        for _ in range(self.num_epochs):
                                # We'll need an "advantage" signal to make PPO work.
                                # We re-compute it at each epoch as its value depends on the value
                                # network which is updated in the inner loop.
                                print("\nSending tensordict data to the advantage module\n")
                                
                        
                                data_view = tensordict_data.reshape(-1)
                                print(f"\nSending reshaped tensordict data of shape {tensordict_data.shape} to  the replay buffer\n")
                                self.replay_buffer.extend(data_view.cpu())
                                for _ in range(self.FRAMES_PER_BATCH // self.sub_batch_size):
                                        
                                        print("\nTraining from sub-batch ")
                                        subdata = self.replay_buffer.sample(self.sub_batch_size)
                                        print(subdata)
                                        quit()
                                        self.advantage_module(tensordict_data)

                                        loss_vals = self.loss_module(subdata.to("cpu"))
                                        loss_value = (
                                                loss_vals["loss_objective"]
                                                + loss_vals["loss_critic"]
                                                + loss_vals["loss_entropy"]
                                        )
                                        print("Made an optimization step")
                                        # Optimization: backward, grad clipping and optimization step
                                        loss_value.backward()
                                        # this is not strictly mandatory but it's good practice to keep
                                        # your gradient norm bounded
                                        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(),1.0)
                                        self.optimizer.step()
                                        self.optimizer.zero_grad()

                        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
                        pbar.update(tensordict_data.numel())
                        cum_reward_str = (
                                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
                        )
                        logs["step_count"].append(tensordict_data["step_count"].max().item())
                        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
                        logs["lr"].append(self.optimizer.param_groups[0]["lr"])
                        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
                        if i % 10 == 0:
                                # We evaluate the policy once every 10 batches of data.
                                # Evaluation is rather simple: execute the policy without exploration
                                # (take the expected value of the action distribution) for a given
                                # number of steps (1000, which is our ``env`` horizon).
                                # The ``rollout`` method of the ``env`` can take a policy as argument:
                                # it will then execute this policy at each step.
                                print("Evaluating policy ")
                                with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                                        # execute a rollout with the trained policy
                                        eval_rollout = self.environment.rollout(1000, self.policy_module)
                                        logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                                        logs["eval reward (sum)"].append(
                                                eval_rollout["next", "reward"].sum().item()
                                        )
                                        logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                                        eval_str = (
                                                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                                                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                                                f"eval step-count: {logs['eval step_count'][-1]}"
                                        )
                                        del eval_rollout
                        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

                        # We're also using a learning rate scheduler. Like the gradient clipping,
                        # this is a nice-to-have but nothing necessary for PPO to work.
                        self.scheduler.step()
            



def ModelRun():
      datapath=os.path.join("data/GBPUSD5min","GBPUSD_M5_2020_01_06_0000_2023_09_04_0045.csv")
      df=pd.read_csv(datapath)[:5000]
      fxbot=ForexModelTrainEval(df)
      fxbot.train()
      
        
if __name__=="__main__":
      ModelRun()
        #algotest(testdata)
        