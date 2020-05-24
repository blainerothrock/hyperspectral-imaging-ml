import torch.nn as nn
import torch.nn.functional as F
import torch
import gin


'''

3d Conv model structure for hyper spectral imaging
- 3 layers of 3d convolutions
- 1 2d convolutional layer
- 2 fully connected layers with dropout layers in between
- output set at 16 for 16 classes but can be made variable for number of feature classes


'''


@gin.configurable()
class HybridSN(nn.Module):
    def __init__(self, num_classes=16, dropout=0.4):
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
            nn.Conv2d(576, 64, kernel_size=3),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(18496, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 3D
        x = self.conv3d(x)

        # 2D
        x = x.view(x.shape[0], x.shape[1] * x.shape[4], x.shape[2], x.shape[3])
        x = self.conv2d(x)

        # Fully-Connected
        x = x.flatten(start_dim=1)
        x = self.linear(x)

        # x = nn.functional.log_softmax(x, 1)
        return x
