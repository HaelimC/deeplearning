"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        self.model = None
        
        self.input_dims = self.hparams["input_dims"]
        self.num_classes = self.hparams["num_classes"]
        self.n_hidden = self.hparams["n_hidden"]
        
        self.lr = self.hparams["learning_rate"]
        self.batch_size=self.hparams["batch_size"]        
        
        self.train_dataset = self.hparams["train_data"]
        self.val_dataset = self.hparams["val_data"]
        self.test_dataset = self.hparams["test_data"]
        
        
               
        
        #conv        
        self.Conv1 = nn.Conv2d(3, 15, 3, padding=1)
        self.Conv2 = nn.Conv2d(15, 30, 3, padding=1)
        self.Conv3 = nn.Conv2d(30, 120, 3, padding=1)
        
        #relu
        self.ReLU = nn.ReLU(inplace=True)
        
        #pool
        self.MaxPool = nn.MaxPool2d(2, stride=2)
        
        #upsampling
        self.UpSample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        #transpose
        self.ConvT3 = nn.ConvTranspose2d(120, 60, 3, padding=1)
        self.ConvT2 = nn.ConvTranspose2d(60, 30, 3, padding=1)
        self.ConvT1 = nn.ConvTranspose2d(30, self.num_classes, 3, padding=1)
        
        #NN
        self.model = nn.Sequential(
            self.Conv1,
            nn.BatchNorm2d(self.n_hidden),
            self.ReLU,
            self.MaxPool,
            self.Conv2,
            nn.BatchNorm2d(self.n_hidden*2),
            self.ReLU,
            self.MaxPool,
            self.Conv3,
            nn.BatchNorm2d(self.n_hidden*2*2*2),
            self.ReLU,
            self.MaxPool,
            self.UpSample,
            self.ConvT3,
            self.ReLU,
            self.UpSample,
            self.ConvT2,
            self.ReLU,
            self.UpSample,
            self.ConvT1,
        )
        
        
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

        x = self.model(x)

        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x
    
    def training_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(out, targets)

        # accuracy
        _, preds = torch.max(out, 1)  # convert output probabilities to predicted class
        acc = preds.eq(targets).sum() / targets.size(0)

        # logs
        #tensorboard_logs = {'loss': loss, 'acc': acc}

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(out, targets)

        # accuracy
        _, preds = torch.max(out, 1)
        acc = preds.eq(targets).sum() / targets.size(0)

        #if batch_idx == 0:
            #self.visualize_predictions(images, out.detach(), targets)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        #tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(out, targets)

        # accuracy
        _, preds = torch.max(out, 1)
        acc = preds.eq(targets).sum() / targets.size(0)

        #if batch_idx == 0:
            #self.visualize_predictions(images, out.detach(), targets)
        return {'test_loss': loss}



    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.lr)
        return optim

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
