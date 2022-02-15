import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


import math

import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import pdb
import numpy as np
import timm



        
class ViT_AvgPool_3modal_CrossAtten_Channel(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_AvgPool_3modal_CrossAtten_Channel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

        #  binary CE
        self.fc = nn.Linear(768, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.drop = nn.Dropout(0.3)
        self.drop2d = nn.Dropout2d(0.3)
        
        # fusion head
        self.ConvFuse = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),    
        )

    def forward(self, x1, x2, x3):

        classtoken1 =  self.vit.forward_features(x1)
        classtoken2 =  self.vit.forward_features(x2)
        classtoken3 =  self.vit.forward_features(x3)
        
        classtoken1 = classtoken1.transpose(1, 2).view(-1, 768, 14, 14)
        classtoken2 = classtoken2.transpose(1, 2).view(-1, 768, 14, 14) 
        classtoken3 = classtoken3.transpose(1, 2).view(-1, 768, 14, 14) 
        
        B,C,H,W = classtoken1.shape
        h1_temp = classtoken1.view(B,C,-1)
        h2_temp = classtoken2.view(B,C,-1)
        h3_temp = classtoken3.view(B,C,-1)
        
        crossh1_h2 = h2_temp @ h1_temp.transpose(-2, -1)    # [64, 2048, 2048]
        #pdb.set_trace()
        crossh1_h2 =F.softmax(crossh1_h2, dim=-1)
        crossedh1_h2 = (crossh1_h2 @ h1_temp).contiguous()  # [64, 2048, 49]
        crossedh1_h2 = crossedh1_h2.view(B,C,H,W)
        
        crossh1_h3 = h3_temp @ h1_temp.transpose(-2, -1)
        crossh1_h3 =F.softmax(crossh1_h3, dim=-1)
        crossedh1_h3 = (crossh1_h3 @ h1_temp).contiguous()
        crossedh1_h3 = crossedh1_h3.view(B,C,H,W)
        
        #h_concat = torch.cat((classtoken1, crossedh1_h2, crossedh1_h3), dim=1)
        h_concat = classtoken1 + crossedh1_h2 + crossedh1_h3
        h_concat = self.ConvFuse(self.drop2d(h_concat))
        
        regmap8 =  self.avgpool8(h_concat)
        
        logits = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))
        
        return logits


