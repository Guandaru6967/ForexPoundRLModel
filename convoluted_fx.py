
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import pytorch_lightning as pl
from scipy.ndimage import shift
from tensordict.nn.distributions import NormalParamExtractor
from torchinfo import summary

class ForexActorNeuralNetwork(nn.Module):
        def __init__(self,num_features,window_size=912):
                super().__init__()
                
        #         self.conv_layers = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # )
                
                self.lstm=nn.LSTM(input_size=num_features,
                                 hidden_size=128, num_layers=4,batch_first=True)

                
                self.policy_option = nn.Sequential(
                       
                        nn.Linear(128,4),
                        nn.Softmax(dim=-1),
                        NormalParamExtractor()
                        )  # Assuming binary option
                self.policy_lot_size = nn.Sequential(
                       
                        nn.Linear(128,4),
                        nn.Sigmoid(dim=-1),
                        NormalParamExtractor()
                        )
                self.policy_tp_size = nn.Sequential(
                       
                        nn.Linear(128,4),
                        nn.Sigmoid(dim=-1),
                        NormalParamExtractor()
                        )
                self.policy_sl_size = nn.Sequential(
                       
                        nn.Linear(128,4),
                        nn.Sigmoid(dim=-1),
                        NormalParamExtractor()
                        )

        def get_conv_output_size(self, input_shape):
                # Function to calculate the output size of the convolutional layers
                test_input = torch.randn(1, *input_shape)
                print(test_input)
                conv_output = self.conv_layers(test_input)
                conv_output_size = conv_output.view(conv_output.size(0), -1).size(-1)
                return conv_output_size
        def forward(self,x:torch.Tensor):
                # x = self.conv_layers(x)
                # print("CNN Ouput shape:",x.shape)
                # # Reshape the output to be compatible with the LSTM input
                # x = x.view(x.size(0), -1, x.size(-2) * x.size(-1))
                print("Input  shape:",x.shape)
                #x = x.view(x.size(0), -1, 32)  # Reshape for LSTM
                # LSTM layer
                _, (x, _)  = self.lstm(x)
                print("Lstm Output:",x.shape)
                # Flatten and pass through heads
               # x = x[:, ]  # Select the output of the last time step
                #x = x.view(x.shape[-1], -1)  # Flatten
                lstm_out_flat = x 
                print("Passing to policy action",lstm_out_flat.shape)
                # Policy heads for each component of the action space
                action_probs_option = nnF.softmax(self.policy_option(lstm_out_flat), dim=-1)
                action_probs_lot_size = torch.sigmoid(self.policy_lot_size(lstm_out_flat))
                action_probs_tp_size = torch.sigmoid(self.policy_tp_size(lstm_out_flat))
                action_probs_sl_size = torch.sigmoid(self.policy_sl_size(lstm_out_flat))

                return (action_probs_option, action_probs_tp_size, action_probs_sl_size,action_probs_lot_size)
                

class ForexCriticNeuralNetwork(nn.Module):
        def __init__(self,num_features,window_size=912):
                super().__init__()
     
                
                self.lstm=nn.LSTM(input_size=num_features,
                                 hidden_size=128, num_layers=4,batch_first=True)

                
                self.value_head=nn.Sequential(
                        nn.Linear(128,4),
                        nn.Softmax(dim=-1),
                        )
        def forward(self,x:torch.Tensor):
                _, (x, _)  = self.lstm(x)
                #print("Lstm Output:",x.shape)
                lstm_out_flat = x[ -1] 
                #print("Passing to policy action",lstm_out_flat.shape)
                # Value head
                state_value = self.value_head(x)
                return state_value

#summary(ForexNeuralNetwork(34,912))
