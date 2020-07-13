import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################


        self.rnn = RNN(input_size, hidden_size, activation)
        self.linear = nn.Linear(hidden_size, classes)
        self.activation = nn.Softmax(dim=2)
        

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    def forward(self, x):
    ############################################################################
    #  TODO: Perform the forward pass                                          #
    ############################################################################   

        h_seq, h = self.rnn(x)
        h = self.linear(h)
        h = self.activation(h)
        x = torch.squeeze(h, 0)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return x


class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
           
        self.lstm = LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, classes)
        self.activation = nn.Softmax(dim=2)


        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################    

        h_seq, (h, c) = self.lstm(x)
        h = self.linear(h)
        h = self.activation(h)
        x = torch.squeeze(h, 0)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return x
