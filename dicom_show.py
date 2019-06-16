# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:58:15 2019

@author: zhang
"""
# =============================================================================
# import os
# import pydicom
# import SimpleITK as sitk
# import cv2
# from matplotlib import pyplot as plt
# 
# if __name__ == '__main__':
#     datapath = r'D:\zyx\AI\data\yangs head\LE_0528\0440.dcm'
#     datapath_HE = r'D:\zyx\AI\data\yangs head\HE_0528\0440.dcm'
# #    datapath = r'D:\OD3DDATA\IMGDATA\20190527\S0000000628\I0000244470.dcm'
# #    data = pydicom.read_file(datapath)
# #    ds = pydicom.dcmread(datapath)
# #    print(ds.PatientName)
# #    print(data.pixel_array.shape)
#     
#     img = sitk.ReadImage(datapath)
#     img_array = sitk.GetArrayFromImage(img)#z, x, y
#     frame_num, width, height = img_array.shape
#     origin = img.GetOrigin() # x, y, z
#     spacing = img.GetSpacing() # x, y, z
#     
#     img_HE = sitk.ReadImage(datapath_HE)
#     img_array_HE = sitk.GetArrayFromImage(img_HE)
#     
#     
#     plt.figure()
#     plt.subplot(131)
#     plt.imshow(img_array[0, :, :],'gray')
#     plt.title('low-energy')
#     plt.subplot(132)
#     plt.imshow(img_array_HE[0, :, :],'gray')
#     plt.title('high-energy')
#     plt.subplot(133)
#     plt.imshow(img_array[0, :, :] - img_array_HE[0, :, :],'gray')
#     plt.title('diff')
# =============================================================================

import visdom
import numpy as np
import os
#os.environ['http_proxy'] = 'http://127.0.0.1:1080'
#os.environ['https_proxy'] = 'http://127.0.0.1:1080'
vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))