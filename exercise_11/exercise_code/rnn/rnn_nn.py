import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        seq_len, batch_size, input_size = x.shape

        h_seq = torch.zeros(seq_len, batch_size, self.hidden_size)
        if h == None:
            h = torch.zeros(1, batch_size, self.hidden_size)
        
        for index in range(seq_len):
            seq = self.linear2(h) + self.linear1(x[index])
            h = self.activation(seq)
            h_seq[index] = h



        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        self.sig3 = nn.Sigmoid()
        self.tan1 = nn.Tanh()
        self.tan2 = nn.Tanh()
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(input_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(input_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.linear7 = nn.Linear(input_size, hidden_size)
        self.linear8 = nn.Linear(hidden_size, hidden_size)



        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []


        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        
        seq_len = x.size(0)
        batch_size = x.size(1)
        
        h_seq = torch.zeros(seq_len, batch_size, self.hidden_size)
        if not h:
            h = torch.zeros(1, batch_size, self.hidden_size)
        if not c:
            c = torch.zeros(1, batch_size, self.hidden_size)
        
        for index in range(seq_len):
            out1 = self.sig1(self.linear1(x[index]) + self.linear2(h))
            out2 = self.sig2(self.linear3(x[index]) + self.linear4(h))
            out3 = self.sig3(self.linear5(x[index]) + self.linear6(h))
            temp = self.tan1(self.linear7(x[index]) + self.linear8(h))
            c = torch.mul(out1,c) + torch.mul(out2, temp)
            h = torch.mul(out3, self.tan2(c))
            h_seq[index] = h



        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        return h_seq , (h, c)

