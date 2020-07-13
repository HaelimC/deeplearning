"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.Conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.Drop1 = nn.Dropout(0.1)
        self.Pool1 = nn.MaxPool2d((2,2))

        self.Conv2 = nn.Conv2d(16, 128, (3, 3))
        self.Drop2 = nn.Dropout(0.3)
        
        self.Conv3 = nn.Conv2d(128, 128, (3, 3))

        self.ConvT3 = nn.ConvTranspose2d(128, 16, (2, 2), stride=(2, 2), padding='same')
        
        self.Conv4 = nn.Conv2d(16, 16, (3, 3))
        self.Drop3 = nn.Dropout(0.1)
        self.Conv5 = nn.Conv2d(16, 16, (3, 3))

        self.Conv6 = nn.Conv2d(16, 1, (1,1))
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        
        
        
        
        x1 = self.Drop1(nn.functional.relu(self.Conv1(x)))
        p1 = self.Pool1((2,2))(x1)
        x2 = nn.functional.relu(self.Conv3(self.Drop2(nn.funtional.relu(Conv2(x1)))))
        t1 = nn.funtional.relu(ConvT3(x2))
        t1 = concatenate([t1, x1], axis=3)
        x3 = nn.functional.relu(self.Conv5(self.Drop3(nn.funtional.relu(Conv4(t1)))))
        x = nn.functional.sigmoid(self.Conv6(x3))
        
        
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
