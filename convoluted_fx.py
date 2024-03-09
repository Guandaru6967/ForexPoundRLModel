from sklearn.preprocessing import  MinMaxScaler,MultiLabelBinarizer,Normalizer
import numpy as np
import pandas as pd
from typing import Tuple
import torchrl
from torchrl.envs import Compose,TransformedEnv,StepCounter,DoubleToFloat,ObservationNorm
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import multiprocessing
import torch.nn as nn
import torch.nn.functional as nnF
from torch.utils.data import random_split
from torchrl.envs import EnvBase
from torch.utils.data import DataLoader,TensorDataset
import pytorch_lightning as pl
from scipy.ndimage import shift
class TimeSeriesPredictionEnv(gym.Env):
    def __init__(self, data, window_size):
        super(TimeSeriesPredictionEnv, self).__init__()
        is_fork = multiprocessing.get_start_method() == "fork"
        self.device = (
                torch.device(0)
                if torch.cuda.is_available() and not is_fork
                else torch.device("cpu")
                )
        self.data = data
        self.window_size = window_size
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.current_step = window_size
        self.max_steps = len(data)

    def reset(self):
        self.current_step = self.window_size
        return self._get_observation()

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}

        obs = self._get_observation()
        reward = self._get_reward(action)
        done = self.current_step == self.max_steps - 1
        self.current_step += 1

        return obs, reward, done, {}

    def _get_observation(self):
        return self.data[self.current_step - self.window_size:self.current_step]

    def _get_reward(self, action):
        # In this example, reward is the absolute difference between the predicted value and the true next value
        predicted_next_value = self.data[self.current_step - 1] + action
        true_next_value = self.data[self.current_step]
        return -np.abs(predicted_next_value - true_next_value)
data = np.sin(np.linspace(0, 100, 2000)) 

tdata=torch.tensor(data)
# plt.plot(data)
# plt.show()

print(tdata)

class LSTMModel(pl.LightningModule):
    def training_step(self,batch,batch_id):
        input,label=batch
        outputs = self.forward(input)
        criterion = nn.MSELoss()
        loss = criterion(outputs,input)
        self.log("loss:",loss)
        return loss
    def configure_optimizers(self) :
         return torch.optim.Adam(self.parameters(), lr=1e-3) 
    def __init__(self,input_size=1,hidden_size=1,output_size=5):
        super().__init__()
        
        self.lstm= nn.LSTM(input_size,hidden_size=hidden_size)
        #self.fc=nn.Linear(in_features=hidden_size,out_features=output_size)
        self.input_size=input_size
        self.output_size=output_size
        pass
    def forward(self,x):
        out,_=self.lstm(x)
        datasample=out
        #out = self.fc(datasample)
        #print("Fully Connected Layer output shape:",out.shape)
        data=out
        #data=nnF.relu(out)
        #print("RELU Output shape:",data.shape)
        return data
datarange=2000
testdata=np.arange(start=0,step=5,stop=datarange)

labeldata=np.arange(start=0,step=1,stop=len(testdata))
model = LSTMModel()
criterion = nn.MSELoss()  # You can use an appropriate loss function for your task

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 


def train():
       
        from sklearn.model_selection import train_test_split
        checkpoint_path = 'model_checkpoint.pth'
        sequences = testdata
        targets = labeldata
        scaler=MinMaxScaler()
        # Convert to PyTorch tensors
        sequences = torch.tensor(sequences, dtype=torch.float32).view(len(sequences),1)
        testingdata=torch.tensor(np.arange(start=datarange,step=5,stop=datarange+5*400))
        print("Testing Data:",testingdata)
        testingdata=torch.tensor(scaler.fit_transform(testingdata.view(len(testingdata),1)),dtype=torch.float32)
        targets = torch.tensor(shift(sequences.view(-1,len(sequences))[0].numpy(), 1, cval=-1), dtype=torch.float32).view(1,len(sequences))
        sequences=torch.tensor(scaler.fit_transform(sequences),dtype=torch.float32).view(len(sequences),1)  # Shape: (num_samples, num_features, sequence_length)
        
        targets=torch.tensor(scaler.fit_transform(targets),dtype=torch.float32).view(len(sequences),1)
        
       
        # Split the dataset into training and testing sets without shuffling
        num_epochs = 5
        
        # print("train_sequences:",train_sequences.shape)
        
        
        epoch=0
        #for epoch in range(num_epochs):
        loss=10
        try:
                loaded_checkpoint=torch.load(checkpoint_path)
                model.load_state_dict(loaded_checkpoint['model_state_dict'])
                optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        except Exception as E:
             print(E)
        
        sampletrain=TensorDataset(sequences,sequences)
        dataloader=DataLoader(dataset=sampletrain,shuffle=True)
        trainer=pl.Trainer(max_epochs=num_epochs,log_every_n_steps=5)
        trainer.fit(model,train_dataloaders=dataloader)
        testvalue=model(sequences)
        testoutput=shift(testvalue.detach().numpy(),1)
        print("Test Output:",torch.tensor(scaler.inverse_transform(torch.tensor(testoutput).view(1,len(testoutput))),dtype=torch.int))
        quit()
        test_sequence=torch.tensor(scaler.fit_transform(torch.tensor(np.arange(start=1655,step=5,stop=1655+5*10)).view(1,10)),dtype=torch.float)#test_sequences[6].view(len(sampletrain),1)
       
        output=model(test_sequence)
        
     
        test_sequence_data=scaler.inverse_transform(test_sequence.view(1,10))
     
        outputdata=torch.cat((torch.tensor(output.detach().numpy()).view(-1,1),torch.zeros((9,1))),dim=0)
       
        output=scaler.inverse_transform(outputdata.view(1,10))
        
        print("Test:",np.round(test_sequence_data))
        answer=output[0][0]
        
        print("Output:",answer)
        print
        while loss>1.5e-1:    
                optimizer.zero_grad()
                outputs = model(sampletrain)
                loss = criterion(outputs,sampletrain)
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
                epoch+=1
        
        torch.save({
        'epoch': 100,  # Add any additional information you want to save
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss  # Add any additional information you want to save
        }, checkpoint_path)
        #print("Test Sequences:",test_sequences,test_sequences.shape)
        #print(test_sequences[1],test_sequences[1].shape)
        # plt.plot(data)
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.title('Time Series Data')
        test_sequence=torch.tensor(scaler.fit_transform(torch.tensor(np.arange(start=1655,step=5,stop=1655+5*10)).view(1,10)),dtype=torch.float)#test_sequences[6].view(len(sampletrain),1)
       
        output=model(test_sequence)
        
     
        test_sequence_data=scaler.inverse_transform(test_sequence.view(1,10))
     
        outputdata=torch.cat((torch.tensor(output.detach().numpy()).view(-1,1),torch.zeros((9,1))),dim=0)
       
        output=scaler.inverse_transform(outputdata.view(1,10))
        
        print("Test:",np.round(test_sequence_data))
        answer=output[0][0]
        
        print("Output:",answer)
        
        plt.show()


train()
# plt.plot(data)

# # Add labels and title
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Time Series Data')
# plt.show()
env=TimeSeriesPredictionEnv(data=data,window_size=20)
transfenv=TransformedEnv(env)

class TimeSeriesPrediction(EnvBase):
     def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
# print("input_spec:", transfenv.input_spec)
# print("action_spec (as defined by input_spec):", transfenv.action_spec)
#print(transfenv.parameters(recurse=False))


# Initialize the MinMaxScaler

