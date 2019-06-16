# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:06:42 2019

@author: zhang
"""
'''
before visdom.Visdom, console 'python -m visdom.server'. port:localhost:8097
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import Unet
import DicomLoader
import visdom
import time

if __name__ == '__main__':
    SavePath = r'D:\zyx\AI\Dicom_process\test.pth'
    LEPath = r'D:\zyx\AI\data\yangs head\img_LE_0102'
    HEPath = r'D:\zyx\AI\data\yangs head\img_HE_0102'
    batch = 1
    Nepoch = 100000
    net = Unet.Unet(3, 2).cuda()    
    testdataset = DicomLoader.testDataset(InPath = r'D:\zyx\data\LE',  #choose one pic for totally overfitting 
                              TargetPath = r'D:\zyx\data\HE',
                              IsPath = True,
                              IsRes = True,
                              transform = transforms.Compose([
                                      DicomLoader.FullyCrop(512),
                                      DicomLoader.ToTensor()]))
    TrainLoader = torch.utils.data.DataLoader(testdataset,
                                              batch_size = batch,
                                              num_workers = 0,
                                              shuffle = True,
                                              pin_memory = False)
    
    net.apply(Unet.InitWeights)
    net.load_state_dict(torch.load(SavePath))
    print('state dict loaded!')
    '''
    loading model
    '''
    criter1 = nn.MSELoss().cuda()
    criter2 = nn.L1Loss().cuda()
    optim = torch.optim.Adam(net.parameters(),
                             lr = 1e-3,
                             betas = (0.9, 0.999),
                             weight_decay = 0)
#    optim = torch.optim.Adam(net.parameters(),
#                             lr = 1e-3,
#                             betas = (0.9, 0.999),
#                             weight_decay = 1e-5)
    MSEVal = np.zeros(Nepoch)
    L1Val = np.zeros(Nepoch)
    vis = visdom.Visdom()
    
    for iepoch in range(Nepoch):
        time_start = time.time()
        net.zero_grad()
        itera = TrainLoader.__iter__()
        for _ in range(len(TrainLoader)):
            optim.zero_grad()
            imageset = itera.next()
            image = imageset['InImg'].cuda()
            target = imageset['TargetImg'].cuda()
            output = net(Variable(image))
            MSELoss = criter1(output, Variable(target))
            L1Loss  = criter2(output, Variable(target))
            MSELoss.backward(retain_graph = True)
            L1Loss.backward()
            optim.step()
            MSEVal[iepoch] += MSELoss.cpu().data.mean()
            L1Val[iepoch] += L1Loss.cpu().data.mean()
        
        MSEVal[iepoch] = MSEVal[iepoch] / len(TrainLoader)
        L1Val[iepoch] = L1Val[iepoch] / len(TrainLoader)
        vis.heatmap(image.cpu().data.numpy()[0,0,:,:].squeeze() + output.cpu().data.numpy()[0,0,:,:].squeeze(),
                    opts = {'colormap' : 'Greys',
                            'title' : 'learned'},
                    win = 1)
        vis.heatmap(image.cpu().data.numpy()[0,0,:,:].squeeze() + target.cpu().data.numpy()[0,0,:,:].squeeze(),
                    opts = {'colormap' : 'Greys',
                            'title' : 'target'},
                    win = 2)
        
        vis.heatmap(output.cpu().data.numpy()[0,0,:,:].squeeze(),
                    opts = {'colormap' : 'Greys',
                            'title' : 'diff'},
                    win = 3)
        
        vis.heatmap(image.cpu().data.numpy()[0,0,:,:].squeeze(),
                    opts = {'colormap' : 'Greys',
                            'title' : 'original'},
                    win = 4)
        
        
        vis.line(MSEVal[0 : iepoch + 1],
                 np.array(range(0, iepoch + 1)),
                 opts = {'title' : 'MSE Loss over time',
                         'xlabel' : 'epoch',
                         'ylabel' : 'loss'},
                 win = 5)
        vis.line(L1Val[0 : iepoch + 1],
                 np.array(range(0, iepoch + 1)),
                 opts = {'title' : 'L1 Loss over time',
                         'xlabel' : 'epoch',
                         'ylabel' : 'loss'},
                 win = 6) 
        time_end = time.time()
        print('Epoch: ' + str(iepoch + 1) + 
              ', MSE loss: ' + str(MSEVal[iepoch]) + 
              ', L1 loss: ' + str(L1Val[iepoch]) + 
              ', elapsed time:' + str((time_end - time_start)))
        torch.save(net.state_dict(), SavePath)
        