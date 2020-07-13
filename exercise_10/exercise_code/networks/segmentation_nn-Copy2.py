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
        #self.double_conv(in_channels, out_channels) = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
        #                                                            nn.ReLU(inplace=True),
        #                                                            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #                                                            nn.ReLU(inplace=True)
        #                                                           )   
        
        self.dconv_down1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, padding=1),
                                         nn.ReLU(inplace=True)
                                         ) #double_conv(3, 64)
        self.dconv_down2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, 3, padding=1),
                                         nn.ReLU(inplace=True)
                                         ) #double_conv(64, 128)
        self.dconv_down3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, 3, padding=1),
                                         nn.ReLU(inplace=True)
                                         ) #double_conv(128, 256)
        #self.dconv_down4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(512, 512, 3, padding=1),
        #                                 nn.ReLU(inplace=True)
        #                                 ) #double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        #self.dconv_up3 = nn.Sequential(nn.Conv2d(256+512, 256, 3, padding=1),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv2d(256, 256, 3, padding=1),
        #                                 nn.ReLU(inplace=True)
        #                                 ) #double_conv(256 + 512, 256)
        self.dconv_up2 = nn.Sequential(nn.Conv2d(128 + 256, 128, 3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, 3, padding=1),
                                         nn.ReLU(inplace=True)
                                         ) #double_conv(128 + 256, 128)
        self.dconv_up1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, padding=1),
                                         nn.ReLU(inplace=True)
                                         ) #double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, num_classes, 1)
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

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        x = self.dconv_down3(x)
        #x = self.maxpool(conv3)   
        
        #x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        
        #x = self.dconv_up3(x)
        #x = self.upsample(x)        
        #x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
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
