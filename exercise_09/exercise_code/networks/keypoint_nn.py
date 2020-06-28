"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        self.Conv1 = nn.Conv2d(1, 32, 5)
        self.Conv2 = nn.Conv2d(32, 64, 4)
        self.Conv3 = nn.Conv2d(64, 128, 3)
        self.Conv4 = nn.Conv2d(128, 256, 2)

        self.Conv1.weight.data *= self.hparams["W1"]
        self.Conv2.weight.data *= self.hparams["W2"]
        self.Conv3.weight.data *= self.hparams["W3"]
        self.Conv4.weight.data *= self.hparams["W4"]

        self.Pool = nn.MaxPool2d(2, 2)
        
        self.Drop1 = nn.Dropout(p=0.1)
        self.Drop2 = nn.Dropout(p=0.2)
        self.Drop3 = nn.Dropout(p=0.3)
        self.Drop4 = nn.Dropout(p=0.4)
        self.Drop5 = nn.Dropout(p=0.5)
        self.Drop6 = nn.Dropout(p=0.6)

        self.Full1 = nn.Linear(4096, 700)
        self.Full2 = nn.Linear(700, 400)
        self.Full3 = nn.Linear(400, 30)


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        if (x.shape == torch.Size([1, 96, 96])):
            x = x.unsqueeze(0)

        x = self.Drop1(self.Pool(nn.functional.relu(self.Conv1(x))))
        x = self.Drop2(self.Pool(nn.functional.relu(self.Conv2(x))))
        x = self.Drop3(self.Pool(nn.functional.relu(self.Conv3(x))))
        x = self.Drop4(self.Pool(nn.functional.relu(self.Conv4(x))))

        x = x.view(x.size()[0], -1)

        x = self.Drop5(nn.functional.relu(self.Full1(x)))
        x = self.Drop6(nn.functional.relu(self.Full2(x)))
        x = self.Full3(x)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
