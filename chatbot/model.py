#importing torch library
import torch
import torch.nn as nn
#importing nn(neural network) from torch library

class NeruralNet(nn.Module):
    def __init__(self, inpt_len, hidd_len, no_cls):
        super(NeruralNet, self).__init__()
        #layer 1
        self.l1 = nn.Linear(inpt_len, hidd_len)
        #layer 2
        self.l2 = nn.Linear(hidd_len, hidd_len)
        #layer 3
        self.l3 = nn.Linear(hidd_len, no_cls)
        #defining the activation function
        self.relu = nn.ReLU()

    # defining a feed forward neural network
    def forward(self, x):
      #pass the value x (vectorized value) to the first layer
        output = self.l1(x)
        # pass the first layer output to the activation function
        output = self.relu(output)
        # pass the output from actionvation function to the 2nd layer
        output = self.l2(output)
        # pass the second layer output to the activation function
        output = self.relu(output)
        # pass the output from actionvation function to the 3nd layer
        output = self.l3(output)

        # return the output
        return output