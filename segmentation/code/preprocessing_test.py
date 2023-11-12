# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:35:00 2022

@author: wama
"""


import numpy as np
import os
import random
import SimpleITK as sitk
import cv2
from skimage import transform
import h5py
import copy


def norm(im):
    
    # mean_ = np.mean(im)
    # std_ = np.std(im)
    # im = (im - mean_) / std_
    max_ = np.max(im)
    min_ = np.min(im)
    im  = (im - min_) / (max_ -  min_)
    im = (im - 0.5)*2

    return im

file_path = '../../data/test/'

for dir, file, images in os.walk(file_path):
    if 'img_raw.nii' in images :
        img_path = dir + '/img_raw.nii.gz'
        img_sitk = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img_sitk).astype(float)
        img = norm(img)
        
        img_resize = np.zeros((img.shape[0],224,224)).astype(float)
        # gt_resize = np.zeros((img.shape[0],224,224)).astype(float)
        for num in range(len(img)):
            img_resize[num] = cv2.resize(img[num],(224,224))
            # gt_resize[num] = cv2.resize(img[num],(224,224),interpolation=cv2.INTER_NEAREST)
        img = copy.copy(img_resize)
        # gt = copy.copy(gt_resize)
        
        gt = np.zeros((img.shape[0],224,224)).astype(float)
        
        f = h5py.File(dir+'/'+'data.npy.h5','w')
        f['image'] = img
        f['label'] = gt
        f.close()
        save_info = dir + '/'
        with open('./lists/test_list.txt','a') as fp:
            fp.write(save_info[2:]+'\n')
            fp.close()
            
        
        
        
        

    
