## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv32 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv64 = nn.Conv2d(32, 64, 3)        
        self.pool2 = nn.MaxPool2d(2,2)


        self.conv128 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.conv256 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2,2)
        
#         self.conv512 = nn.Conv2d(256, 512, 1)
#         self.pool5 = nn.MaxPool2d(2,2)
#         6 6 512
        self.dense1 = nn.Linear(12*12*256, 1024)
        self.do1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(1024, 1024)
        self.do2 = nn.Dropout(p=0.3)
        self.dense3 = nn.Linear(1024, 1024)
        self.do3 = nn.Dropout(p=0.4)
        self.dense4 = nn.Linear(1024, 1024)
        self.do4 = nn.Dropout(p=0.5)
        self.dense5 = nn.Linear(1024,136)
    

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))          # layer shape in: 224,224,3
        
        x = self.pool1(F.selu(self.conv32 (x)))
        x = self.pool2(F.selu(self.conv64 (x)))
        x = self.pool3(F.selu(self.conv128(x)))
        x = self.pool4(F.selu(self.conv256(x)))
#         x = self.pool5(F.selu(self.conv512(x)))
        
        x = x.view(x.size(0), -1)
        
        x = self.do1(F.selu(self.dense1(x)))
        x = self.do2(F.selu(self.dense2(x)))
        x = self.do3(F.selu(self.dense3(x)))
        x = self.do4(F.selu(self.dense4(x)))
        x = self.dense5(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
