import torch.nn as nn
import torch.nn.functional as F
import torch


'''

3d Conv model structure for hyper spectral imaging
- 3 layers of 3d convolutions
- 1 2d convolutional layer
- 2 fully connected layers with dropout layers in between
- output set at 16 for 16 classes but can be made variable for number of feature classes


'''


class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 7)),
            nn.ReLU(),
            nn.Conv3d(8, 16, (3, 3, 5)),
            nn.ReLU(),
            nn.Conv3d(16, 32, (3, 3, 3)),
            nn.ReLU()
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(576, 64, (3, 3)),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(1156,16),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(16, 1),
            nn.Dropout(p=0.4),
            nn.ReLU()
        )



    def forward(self, x):


        x = self.conv3d(x)

        x = x.reshape((-1, x.shape[1]* x.shape[4],x.shape[2] , x.shape[3]))
        x = self.conv2d(x)
        x = torch.flatten(x)
        x = x.view(-1, 16)
        x = x.reshape(x.shape[1],x.shape[0])
        x = self.linear(x)
        #final change might be unnecessary because it change in output shape --> same tensors
        x = x.view(-1, 16)

        return x[0]
