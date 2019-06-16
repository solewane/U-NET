# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:33:01 2019

@author: zhang
"""
'''
Convolution layers, or the left part of U-net, is feature-capturer, like resnet/
vgg/inception etc.
So that pretrained model makes sense. Transfer training is of great significance!
||U-net = resnet + upsampling||   
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def InitWeights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight.data)
        if layer.bias is True:
            nn.init.xavier_normal_(layer.bias.data)
        
class UBlock(nn.Module):
    def __init__(self, KernelSize, InChn, OutChn):
        super(UBlock, self).__init__()
        self.conv1 = nn.Conv2d(InChn, OutChn, KernelSize, padding = 1,bias = True)
        self.bn1 = nn.BatchNorm2d(OutChn)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(OutChn, OutChn, KernelSize, padding = 1, bias = True)
        self.bn2 = nn.BatchNorm2d(OutChn)
        self.act2 = nn.Tanh()
        
    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.act1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.bn2(inputs)
        inputs = self.act2(inputs)
        return inputs
    

class Unet(nn.Module):
    def __init__(self, KernelSize, SampleRate):
        super(Unet, self).__init__()
        self.SampleRate = SampleRate
        '''Encoder'''
        self.encoder1 = UBlock(KernelSize, 1, 16)
        self.downsam1 = nn.MaxPool2d(SampleRate)
        self.encoder2 = UBlock(KernelSize, 16, 32)
        self.downsam2 = nn.MaxPool2d(SampleRate)        
        self.encoder3 = UBlock(KernelSize, 32, 64)
        self.downsam3 = nn.MaxPool2d(SampleRate)        
        self.encoder4 = UBlock(KernelSize, 64, 128)
        self.downsam4 = nn.MaxPool2d(SampleRate)
        self.encoder5 = UBlock(KernelSize, 128, 256)
        '''Decoder'''
        '''LIU uses 1x1 conv2d rather than ConvTranspose2d'''
#        self.upsam5 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
#        self.decoder5 = UBlock(KernelSize, 256, 128)
#        self.upsam4 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
#        self.decoder4 = UBlock(KernelSize, 128, 64)
#        self.upsam3 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
#        self.decoder3 = UBlock(KernelSize, 64, 32)
#        self.upsam2 = nn.ConvTranspose2d(32, 16, 2, stride = 2)
#        self.decoder2 = UBlock(KernelSize, 32, 16)
#        self.decoder1 = nn.Conv2d(16, 1, 1)
        self.upsam5 = nn.Conv2d(256, 128, 1)
        self.decoder5 = UBlock(KernelSize, 256, 128)
        self.upsam4 = nn.Conv2d(128, 64, 1)
        self.decoder4 = UBlock(KernelSize, 128, 64)
        self.upsam3 = nn.Conv2d(64, 32, 1)
        self.decoder3 = UBlock(KernelSize, 64, 32)
        self.upsam2 = nn.Conv2d(32, 16, 1)
        self.decoder2 = UBlock(KernelSize, 32, 16)
        self.decoder1 = nn.Conv2d(16, 1, 1)
        
    def forward(self, inputs):
        inputs = self.encoder1(inputs)
        temp1 = inputs#[:, :, 88:-88, 88:-88]
        inputs = self.downsam1(inputs)
        inputs = self.encoder2(inputs)
        temp2 = inputs#[:, :, 40:-40, 40:-40]
        inputs = self.downsam2(inputs)
        inputs = self.encoder3(inputs)
        temp3 = inputs#[:, :, 16:-16, 16:-16]
        inputs = self.downsam3(inputs)
        inputs = self.encoder4(inputs)
        temp4 = inputs#[:, :, 4:-4, 4:-4]
        inputs = self.downsam4(inputs)
        inputs = self.encoder5(inputs)  
        '''Decoder'''
        inputs = torch.nn.functional.interpolate(inputs, scale_factor = self.SampleRate, mode = 'bilinear')
        inputs = self.upsam5(inputs)
        inputs = torch.cat((temp4, inputs), dim = 1)
        inputs = self.decoder5(inputs)        
        inputs = torch.nn.functional.interpolate(inputs, scale_factor = self.SampleRate, mode = 'bilinear')
        inputs = self.upsam4(inputs)
        inputs = torch.cat((temp3, inputs), dim = 1)
        inputs = self.decoder4(inputs) 
        inputs = torch.nn.functional.interpolate(inputs, scale_factor = self.SampleRate, mode = 'bilinear')
        inputs = self.upsam3(inputs)
        inputs = torch.cat((temp2, inputs), dim = 1)
        inputs = self.decoder3(inputs) 
        inputs = torch.nn.functional.interpolate(inputs, scale_factor = self.SampleRate, mode = 'bilinear')
        inputs = self.upsam2(inputs)
        inputs = torch.cat((temp1, inputs), dim = 1)
        inputs = self.decoder2(inputs) 
        inputs = self.decoder1(inputs)
        return inputs
    
    
if __name__ == '__main__':
#    __spec__ = None
    inputs = torch.FloatTensor(1, 1, 512, 512)
    net = Unet(3, 2)
    outputs = net(Variable(inputs))