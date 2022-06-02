import torch as T
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
# import numpy as np
import time

class autoencoder(nn.Module):
    def __init__(self, channels, outputs): #, learning_rate, tb_dir, models_dir, weights=None):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 64, 4, 1)
        
        self.conv7 = nn.ConvTranspose2d(64, 512, 4, 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv11 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn11 = nn.BatchNorm2d(32)
        self.conv12 = nn.ConvTranspose2d(32, outputs, 4, 2, 1)
        '''
        if weights is not None:
            self.loss = nn.CrossEntropyLoss(weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimiser = optim.Adam(self.parameters(), self.learning_rate)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.tb_dir = tb_dir
        self.models_dir = models_dir
        self.tensorboard = SummaryWriter(self.tb_dir + f'/ae-{int(time.time())}')
        self.to(self.device)
        '''
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv12(x)
        x = T.sigmoid(x)
        
        return x
    
    def compress(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.conv6(x)
        
        return x
    
    def save_model(self, r, t, ep):
        T.save(self.state_dict(), self.models_dir + f"/ae-{int(time.time())}")