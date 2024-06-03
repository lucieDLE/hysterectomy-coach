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

import pytorchvideo.models.resnet
from pytorchvideo.models.head import ResNetBasicHead


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
            
        self.loss = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)
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
            nn.Linear(in_features=256, out_features=out_features, bias=True),
            nn.Dropout(p=0.2, inplace=True)
            )
        

        ###################### 
        self.softmax = nn.Softmax(dim=1)

        ######################
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
        x = self.softmax(x,dim=1)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.model(x)

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

class ResNetLSTM(pl.LightningModule):
    def __init__(self, args=None, out_features=4, class_weights=None, features=False):
        super(ResNetLSTM, self).__init__()
        self.save_hyperparameters()
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)

        self.loss = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())[:-1]


        self.model_dict = nn.ModuleDict({
            'resnet':nn.Sequential(*layers),
            'lstm': nn.LSTM(2048, 512, batch_first=True),
            'fc_c': nn.Linear(512, out_features),
            'dropout': nn.Dropout(p=0.2)
        })

        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][0])
        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][1])
        init.xavier_uniform_(self.model_dict['fc_c'].weight)


    def configure_optimizers(self):
        optimizer = optim.SGD([
                {'params': self.model_dict['resnet'].parameters()},
                {'params': self.model_dict['lstm'].parameters()},
                {'params': self.model_dict['fc_c'].parameters(), 'lr': self.args.lr},
                ], lr=self.args.lr / 10, momentum=0.9, dampening=0, weight_decay=5e-4, nesterov=False)

        return optimizer
    
    def forward(self, x):
        x = x.contiguous()
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
    
        x = self.model_dict['resnet'](x)
        x = x.view(-1, num_frames, 2048)

        self.model_dict['lstm'].flatten_parameters()
        y, _ = self.model_dict['lstm'](x)
        y = y.contiguous().view(-1, 512)
        y = self.model_dict['dropout'](y)

        return y
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.forward(x)
        x = self.model_dict['fc_c'](x)

        x = x[self.args.num_frames - 1::self.args.num_frames]

        loss = self.loss(x, y)
        self.log('train_loss', loss.item(), sync_dist=True)

        x = torch.argmax(x, dim=1)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)


        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        x = self.forward(x)
        x = self.model_dict['fc_c'](x)

        x = x[self.args.num_frames - 1::self.args.num_frames]

        loss = self.loss(x, y)
        self.log('val_loss', loss.item(), sync_dist=True, )


        x = torch.argmax(x, dim=1)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

        return loss

class ResNetLSTM_p2(pl.LightningModule):
    def __init__(self, args=None, out_features=4, class_weights=None, features=False):
        super(ResNetLSTM_p2, self).__init__()

        self.save_hyperparameters()
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        self.memory_model = ResNetLSTM.load_from_checkpoint(self.args.lfb_model, strict=True)
        self.memory_model.eval()
        self.memory_model.cuda()
        self.length_vid = 10

            
        self.loss = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())[:-1]

        self.model_dict = nn.ModuleDict({
            'memory': self.memory_model,
            'resnet':nn.Sequential(*layers),
            'lstm': nn.LSTM(2048, 512, batch_first=True),
            'fc_c': nn.Linear(512, out_features),
            'fc_h_c': nn.Linear(1024, 512),
            'nl_bloc': NLBlock(),
            'time_conv': TimeConv(),
            'dropout': nn.Dropout(p=0.5)
        })

        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][0])
        init.xavier_normal_(self.model_dict['lstm'].all_weights[0][1])
        init.xavier_uniform_(self.model_dict['fc_c'].weight)
        init.xavier_uniform_(self.model_dict['fc_h_c'].weight)

    def forward(self, x, long_feature):

        batch_size, num_frames, channels, height, width = x.size()
        x = x.contiguous().view(batch_size * num_frames, channels, height, width)
    
        ## creating features
        x = self.model_dict['resnet'](x)
        x = x.view(-1, num_frames, 2048)
        self.model_dict['lstm'].flatten_parameters()
        y, _ = self.model_dict['lstm'](x)
        y = y.contiguous().view(-1, 512)
        y = self.model_dict['dropout'](y)

        y = y[num_frames - 1::num_frames]

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

        long_features= self.model_dict['memory'](x) # (BS* num_frames, 512)
        long_features = long_features.view(-1, self.args.num_frames, 512)

        ## take only 10 frames in x 
        x = x[:,-self.length_vid:,:,:,:]

        x = self.forward(x, long_features)

        loss = self.loss(x, y)
        self.log('train_loss', loss, sync_dist=True)

        x = torch.argmax(x, dim=1)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        long_features= self.model_dict['memory'](x) # (BS* num_frames, 512)
        long_features = long_features.view(-1, self.args.num_frames, 512)

        ## take only 10 frames in x 
        x = x[:,-self.length_vid:,:,:,:]

        x = self.forward(x, long_features)

        loss = self.loss(x, y)
        self.log('val_loss', loss.item(), sync_dist=True)

        x = torch.argmax(x, dim=1)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

        return loss

class pytorch3DResnet(pl.LightningModule):
    def __init__(self, args= None, out_features=4, class_weights=None, features=False):
        super(pytorch3DResnet, self).__init__()

        self.save_hyperparameters()
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        self.loss = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)


        self.backbone = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
        layers = list(self.backbone.blocks)[:-1]

        # replace last layer with same layer but with out_feature as output size
        resnet_head = ResNetBasicHead(dropout=nn.Dropout(p=0.5),
                                      proj=nn.Linear(in_features=2304, out_features=out_features),
                                      output_pool=torch.nn.AdaptiveAvgPool3d(output_size=1))

        self.model_dict = nn.ModuleDict({
            'slowfast':nn.Sequential(*layers),
            'resnet_head': resnet_head,
            })

    def forward(self, x):
        # pdb.set_trace()
        y = self.model_dict['slowfast'](x)
        y = self.model_dict['resnet_head'](y)

        return y

    def training_step(self, batch, batch_idx):
        # print(batch.type())
        x,y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item())


        y_hat = torch.argmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        # print(batch.type())
        x,y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        self.log("val_loss", loss.item())

        y_hat = torch.argmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        self.log("val_acc", self.accuracy)

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-1)