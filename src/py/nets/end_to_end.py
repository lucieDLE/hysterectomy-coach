import math
import numpy as np 
import pdb

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import overload
from torchvision import models
from torchvision import transforms
import torchmetrics
import torch.optim as optim

import pytorch_lightning as pl
import torch.nn.init as init

from sklearn.metrics import classification_report
# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class NLBlock(nn.Module):
    def __init__(self, feature_num=512):
        super(NLBlock, self).__init__()
        self.linear1 = nn.Linear(feature_num, feature_num)
        self.linear2 = nn.Linear(feature_num, feature_num)
        self.linear3 = nn.Linear(feature_num, feature_num)
        self.linear4 = nn.Linear(feature_num, feature_num)
        self.layer_norm = nn.LayerNorm([1, 512])
        self.dropout = nn.Dropout(0.2)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)
        init.xavier_uniform_(self.linear4.weight)

    def forward(self, St, Lt):
        St_1 = St.view(-1, 1, 512)
        St_1 = self.linear1(St_1)
        Lt_1 = self.linear2(Lt)
        Lt_1 = Lt_1.transpose(1, 2)
        SL = torch.matmul(St_1, Lt_1)       
        SL = SL * ((1/512)**0.5)
        SL = F.softmax(SL, dim=2)
        Lt_2 = self.linear3(Lt)
        SLL = torch.matmul(SL, Lt_2)
        SLL = self.layer_norm(SLL)
        SLL = F.relu(SLL)
        SLL = self.linear4(SLL)
        SLL = self.dropout(SLL)
        SLL = SLL.view(-1, 512)
        return (St+SLL)
    
class TimeConv(nn.Module):
    def __init__(self):
        super(TimeConv, self).__init__()
        self.timeconv1 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.timeconv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.timeconv3 = nn.Conv1d(512, 512, kernel_size=7, padding=3)
        self.maxpool_m = nn.MaxPool1d(2, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d((512,1))

    def forward(self, x):
        x = x.transpose(1, 2)
        
        x1 = self.timeconv1(x)
        y1 = x1.transpose(1, 2)
        y1 = y1.view(-1,x.shape[2],512,1)

        x2 = self.timeconv2(x)
        y2 = x2.transpose(1, 2)
        y2 = y2.view(-1,x.shape[2],512,1)

        x3 = self.timeconv3(x)
        y3 = x3.transpose(1, 2)
        y3 = y3.view(-1,x.shape[2],512,1)

        x4 = F.pad(x, (1,0), mode='constant', value=0)
        x4 = self.maxpool_m(x4)
        y4 = x4.transpose(1, 2)
        y4 = y4.view(-1,x.shape[2],512,1)

        y0 = x.transpose(1, 2)
        y0 = y0.view(-1,x.shape[2],512,1)

        y = torch.cat((y0,y1,y2,y3,y4), dim=3)
        y = self.maxpool(y)
        y = y.view(-1,x.shape[2],512)
        
        return y



class MemoryBank(nn.Module):
    def __init__(self, args=None):
        super(MemoryBank, self).__init__()

        self.args = args

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())[:-1]

        self.model_dict = nn.ModuleDict({
            'resnet':nn.Sequential(*layers),
            # 'attention': nn.MultiheadAttention(2048, num_heads=8),
            'lstm': nn.LSTM(2048, 512, batch_first=True),
            'dropout': nn.Dropout(p=0.2),
        })

        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][0])
        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][1])
    
    def forward(self, x):

        x = x.contiguous()
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
    
        x = self.model_dict['resnet'](x)
        x = x.view(-1, num_frames, 2048)

        # x, _ = self.model_dict['attention'](x,x,x)
        
        self.model_dict['lstm'].flatten_parameters()
        y, _ = self.model_dict['lstm'](x)
        y = y.contiguous().view(-1, 512)
        y = self.model_dict['dropout'](y)

        return y
    

class ResNetLSTM(pl.LightningModule):
    def __init__(self, args=None, out_features=4, class_weights=None, features=False):
        super(ResNetLSTM, self).__init__()

        self.save_hyperparameters()
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

            
        self.loss = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        layers = list(backbone.children())[:-1]
        self.softmax = nn.Softmax(dim=1)


        self.model_dict = nn.ModuleDict({
            'memory': MemoryBank(),
            'resnet':nn.Sequential(*layers),
            # 'attention': nn.MultiheadAttention(2048, num_heads=8),
            'lstm': nn.LSTM(2048, 512, batch_first=True),
            'fc_c': nn.Linear(512, out_features),
            'fc_h_c': nn.Linear(1024, 512),
            'nl_bloc': NLBlock(),
            'time_conv': TimeConv(),
            'dropout': nn.Dropout(p=0.5),
        })

        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][0])
        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][1])
        init.xavier_uniform_(self.model_dict['fc_c'].weight)
        init.xavier_uniform_(self.model_dict['fc_h_c'].weight)

    def forward(self, x):

        sequence_length = 1

        ## compute memory
        batch_size, num_frames, channels, height, width = x.size()
        long_feature = self.model_dict['memory'](x)
        long_feature = long_feature.view(-1, num_frames, 512)

        x = x[:,-sequence_length:,:,:,:]

        x = x.contiguous()
        x = x.view(batch_size * sequence_length, channels, height, width)

        # forward
        x = self.model_dict['resnet'](x)
        x = x.view(-1, sequence_length, 2048)
        # x, _  = self.model_dict['attention'](x,x,x)

        self.model_dict['lstm'].flatten_parameters()
        y, _ = self.model_dict['lstm'](x)
        y = y.contiguous().view(-1, 512)
        y = self.model_dict['dropout'](y)

        y = y[sequence_length - 1::sequence_length]

        Lt = self.model_dict['time_conv'](long_feature)

        y_1 = self.model_dict['nl_bloc'](y, Lt)
        y = torch.cat([y, y_1], dim=1)
        y = self.model_dict['dropout'](self.model_dict['fc_h_c'](y))
        y = F.relu(y)
        y = self.model_dict['fc_c'](y)
        
        return y

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = optim.SGD([
                {'params': self.model_dict['resnet'].parameters()},
                {'params': self.model_dict['lstm'].parameters()},
                {'params': self.model_dict['time_conv'].parameters(), 'lr': lr},
                {'params': self.model_dict['nl_bloc'].parameters(), 'lr': lr},
                {'params': self.model_dict['fc_h_c'].parameters(), 'lr': lr},
                {'params': self.model_dict['fc_c'].parameters(), 'lr': lr},
                ], lr=lr / 10, momentum=0.9, dampening=0, weight_decay=5e-4, nesterov=False)

        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = self.forward(x)

        loss = self.loss(x, y)
        self.log('train_loss', loss, sync_dist=True)

        x = torch.argmax(x, dim=1)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x = self.forward(x)

        loss = self.loss(x, y)
        self.log('val_loss', loss, sync_dist=True)

        x = torch.argmax(x, dim=1)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

        return loss