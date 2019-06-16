# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:12:19 2019

@author: zhang
"""
'''
load all training data which is a huge matrix(H, W, N) in memory is expensive,
an alternative is to store the path or URL which is a string list(N, ), single
datum(H, W)  is loaded in training processs steps.
'''
import os
import glob
import numpy as np
import SimpleITK as sitk
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

def DicomReader(dicompath):
    img = sitk.ReadImage(dicompath)
    img_array = sitk.GetArrayFromImage(img) #z, x, y
    return img_array[0, :, :] # int16 numpy

def ListDicomData(LowDcmPath, HighDcmPath, IsReturnPath = False):
    print('Data loading...\n')
    time_start = time.time()
    LowDcmDatas = []
    HighDcmDatas = []
    FolderCount, DcmCount = 0, 0
    for dirpath, dirnames, filenames in os.walk(LowDcmPath):
        if dirpath == LowDcmPath:
            foldernum = max(len(dirnames), 1)
            print('%d filedir pairs' %(foldernum))
        if len(dirnames) == 0:
            FolderCount += 1
            dcmfiles = glob.glob(os.path.join(dirpath, '*.dcm'))
            temp = DicomReader(dcmfiles[0])
            Height, Width = temp.shape
            if IsReturnPath:
                LowDcmData = []
                HighDcmData = []            
                for i in range(len(dcmfiles)):
                    LowDcmData.append(dcmfiles[i])
                    HighDcmData.append(dcmfiles[i].replace(LowDcmPath, HighDcmPath))
            else:            
                LowDcmData = np.zeros((Height, Width, len(dcmfiles)))
                HighDcmData = np.zeros((Height, Width, len(dcmfiles)))
                for i in range(len(dcmfiles)):
                    LowDcmData[:, :, i] = DicomReader(dcmfiles[i])
                    HighDcmData[:, :, i] = DicomReader(dcmfiles[i].replace(LowDcmPath, HighDcmPath))
            LowDcmDatas.append(LowDcmData)
            HighDcmDatas.append(HighDcmData)
            DcmCount += len(dcmfiles)
            print('%d dicom pairs loaded successfully, folder %d/%d' %(len(dcmfiles), FolderCount, foldernum))
    time_end = time.time()
    print('%d dicom pairs in total, elapsed time: %f seconds' % (DcmCount, (time_end - time_start)))
    return(LowDcmDatas, HighDcmDatas, DcmCount)
    
def DicomLoader(LowDcmPath, HighDcmPath):
    print('Data loading...\n')
    time_start = time.time()
    LowDcmDatas = []
    HighDcmDatas = []
    for filename in os.listdir(LowDcmPath):
        if filename.endswith('.dcm'):
            LowDcmFile = os.path.join(LowDcmPath, filename)
            LowDcmData = DicomReader(LowDcmFile) 
            LowDcmDatas.append(LowDcmData)
            HighDcmFile = os.path.join(HighDcmPath, filename)
            assert os.path.exists(HighDcmFile)
            HighDcmData = DicomReader(HighDcmFile)
            HighDcmDatas.append(HighDcmData)
    time_end = time.time()
    print('%d dicom pairs loaded successfully, elapsed time: %f seconds' % (len(LowDcmDatas), (time_end - time_start)))
    return(LowDcmDatas, HighDcmDatas)

def DicomPathLoader(LowDcmPath, HighDcmPath):
    print('Data loading...\n')
    time_start = time.time()
    LowDcmDatas = []
    HighDcmDatas = []
    for filename in os.listdir(LowDcmPath):
        if filename.endswith('.dcm'):
            LowDcmFile = os.path.join(LowDcmPath, filename)
#            LowDcmData = DicomReader(LowDcmFile) 
            LowDcmDatas.append(LowDcmFile)
            HighDcmFile = os.path.join(HighDcmPath, filename)
            assert os.path.exists(HighDcmFile)
#                HighDcmData = DicomReader(HighDcmFile)
            HighDcmDatas.append(HighDcmFile)
    time_end = time.time()
    print('%d dicom pairs loaded successfully, elapsed time: %f seconds' % (len(LowDcmDatas), (time_end - time_start)))
    return(LowDcmDatas, HighDcmDatas)

class ToTensor(object):
    def __init__(self, Mean = 0, Var = 1):
        self.Mean = Mean
        self.Var = Var
        
    def __call__(self, Img):
        OutImg = (Img - self.Mean) / self.Var
        return torch.from_numpy(OutImg)
    
class FullyCrop(object):
    def __init__(self, ImgSize):
        self.ImgSize = ImgSize
        
        
    def __call__(self, Img):
        _, H, W = Img.shape
        offset = (H - self.ImgSize) // 2
        return Img[:, offset : offset + self.ImgSize, offset : offset + self.ImgSize]
        
#    def __call__(self, sample):
#        InImg, TargetImg = sample['InImg'], sample['TargetImg']
#        InImg = (InImg - self.Mean) / self.Var
#        TargetImg = (TargetImg - self.Mean) / self.Var
#        return {'InImg': torch.from_numpy(InImg)
#                'TargetImg': torch.from_numpy(TargetImg)}
       
class testDataset(Dataset):
    '''torch.utils.data.Dataset : virtual base class'''  
    def __init__(self, InPath, TargetPath, IsPath = False, IsRes = False, transform = None):
        Input, Target, SampleNum = ListDicomData(InPath, TargetPath, IsReturnPath = IsPath)
        '''IsPath is a Boolean value used to indicate whether the inputs are paths'''
        if IsPath:
            self.InData = []
            self.TargetData = []
            for idx in range(len(Input)):
                self.InData.extend(Input[idx])
                self.TargetData.extend(Target[idx])
        else:
            startIdx = 0              
            self.InData = np.zeros((Input[0].shape[0], Input[0].shape[1], SampleNum), 'float32')
            self.TargetData = np.zeros((Input[0].shape[0], Input[0].shape[1], SampleNum), 'float32')
            for idx in range(len(Input)):
                endIdx = startIdx + Input[idx].shape[2]
                self.InData[:, :, startIdx:endIdx] = Input[idx]
                self.TargetData[:, :, startIdx:endIdx] = Target[idx]
                startIdx = endIdx
        self.SampleNum = SampleNum
        self.IsPath = IsPath
        self.IsRes = IsRes
        self.transform = transform
        
    def __len__(self):
        return self.SampleNum
        
    def __getitem__(self, idx):  
        '''load : path-matrix or directly matrix? which is memory-friendly?'''
        if self.IsPath:
            InData = self.InData[idx]
            TargetData = self.TargetData[idx]
            InData = DicomReader(InData)
            TargetData = DicomReader(TargetData)
            InImg = np.copy(np.reshape(InData,(1, InData.shape[0], InData.shape[1]))).astype('float32')
            TargetImg = np.copy(np.reshape(TargetData,(1, TargetData.shape[0], TargetData.shape[1]))).astype('float32')
        else:
            InData = self.InData[:, :, idx]
            TargetData = self.TargetData[:, :, idx]
            InImg = np.copy(np.reshape(InData,(1, InData.shape[0], InData.shape[1])))
            TargetImg = np.copy(np.reshape(TargetData,(1, TargetData.shape[0], TargetData.shape[1])))
        '''the target to learn: residual or image'''
        if self.IsRes:
            TargetImg = TargetImg - InImg
        if self.transform:
            InImg = self.transform(InImg)       
            TargetImg = self.transform(TargetImg)
        sample = {'InImg':InImg, 'TargetImg':TargetImg}
        return sample
        
        
if __name__ == '__main__':
    
    testdataset = testDataset(InPath = r'D:\zyx\AI\data\yangs head\img_LE_0102',
                              TargetPath = r'D:\zyx\AI\data\yangs head\img_HE_0102',
                              IsPath = True,
                              IsRes = True,
                              transform = transforms.ToTensor())
    '''ToTensor:(1, 702, 702)->(702, 1, 702)??'''
    '''CauZ: img = torch.from_numpy(pic.transpose((2, 0, 1)))'''
    print(len(testdataset))
    sample = testdataset[100]
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(sample['InImg'][0, :, :],'gray')
    plt.title('low-energy')
    plt.subplot(132)
    plt.imshow(sample['TargetImg'][0, :, :],'gray')
    plt.title('high-energy')
    plt.subplot(133)
    plt.imshow(sample['InImg'][0, :, :] - sample['TargetImg'][0, :, :],'gray')
    plt.title('diff')
    
#    LowDcmDatas, HighDcmDatas, _ = ListDicomData(r'D:\zyx\AI\data\yangs head\img_LE_0102', 
#                                                 r'D:\zyx\AI\data\yangs head\img_HE_0102',
#                                                 IsReturnPath = True)


#    fileDir = r'D:\zyx\AI\data\yangs head\img_HE_0102'
#    for root, dirs, files in os.walk(fileDir):  
##    #begin  
#        print(root)  
#        print(dirs)  
#        print(files)  