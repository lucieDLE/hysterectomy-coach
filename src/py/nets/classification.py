import math
import numpy as np 

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
import torchmetrics

import pytorch_lightning as pl


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class RecurrentLayer(nn.Module):
    def __init__(self, in_features=256, out_features=128, num_layers=2):
        super(RecurrentLayer, self).__init__()
        self.model = nn.LSTM(in_features, out_features, num_layers, batch_first=True)
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        output, (hn, cn) = self.model(input_seq)

        return torch.cat(list(cn), dim=-1)

class HystNet(pl.LightningModule):
    def __init__(self, args = None, out_features=4, class_weights=None, features=False):
        super(HystNet, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)

        
        # feat = models.efficientnet_v2_s(weights=models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT)
        # feat.classifier = nn.Identity()
        # num_feat = 1280
        feat = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        feat.fc = nn.Identity()
        num_feat = 512

        self.model = torch.nn.Sequential(
            TimeDistributed(feat), 
            nn.Linear(in_features=num_feat, out_features=256, bias=True), 
            RecurrentLayer(in_features=256, out_features=128, num_layers=2),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=out_features, bias=True)
            )        
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transforms = TimeDistributed(nn.Sequential(                
                transforms.RandomApply([transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=90),
            ))
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):                
        x = self.model(x)
        x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.model(self.train_transforms(x))

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss.item(), sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)