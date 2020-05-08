# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm


class model(nn.Module):
    def __init__(self):
#         super(model, self).__init__()
        # input 1 (InputLayer) (25, 25, 30, 1) 0
        # conv3d 1 (Conv3D) (23, 23, 24, 8) 512
        # conv3d 2 (Conv3D) (21, 21, 20, 16) 5776
        # conv3d 3 (Conv3D) (19, 19, 18, 32) 13856
        # reshape 1 (Reshape) (19, 19, 576) 0
        # conv2d 1 (Conv2D) (17, 17, 64) 331840
        # flatten 1 (Flatten) (18496) 0
        # dense 1 (Dense) (256) 4735232
        # dropout 1 (Dropout) (256) 0
        # dense 2 (Dense) (128) 32896
        # dropout 2 (Dropout) (128) 0
        # dense 3 (Dense) (16) 2064
        #first 3d layer
        self.conv1 = nn.Conv3d(25,8, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        #second 3d layer
        self.conv2 = nn.Conv3d(8, 8,(3, 3, 3),stride=(1, 1, 1), padding=(0,1,1))
        #third 3d layer
        self.conv3 = nn.Conv3d(8, 16, 1,stride=1)
        #first 2d layer
        self.conv4 = nn.Conv2d(30, 16, 1, stride=(1))
        #first fully connected dense layer
        self.linear = nn.Linear(2560,16)
        #second fully connected dense layer
        self.linear2 = nn.Linear(16,16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv3d_shape = x.shape
         #reshape 3d layer for 2d
        x = output3.reshape((conv3d_shape[1], conv3d_shape[2]*conv3d_shape[0], conv3d_shape[3]*conv3d_shape[4],1))
        x = F.relu(self.conv4(x))
        #flatten 2d for fully dense layers
        x = torch.flatten(x)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        #shape output for predicted classes assumed 16 from dataset
        x = x.view(-1, 16)
        return x[0]
