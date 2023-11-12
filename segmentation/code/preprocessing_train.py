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

file_path = '../../data/train_seg/'

for dir, file, images in os.walk(file_path):
    if 'img_raw.nii' in images :
        img_path = dir + '/img_raw.nii'
        img_sitk = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img_sitk)
        img = norm(img)
        
        gt_path = dir + '/gt.nii'
        gt_sitk = sitk.ReadImage(gt_path)
        gt = sitk.GetArrayFromImage(gt_sitk)
        
        img_resize = np.zeros((img.shape[0],224,224))
        gt_resize = np.zeros((img.shape[0],224,224))
        for num in range(len(img)):
            img_resize[num] = cv2.resize(img[num],(224,224))
            gt_resize[num] = cv2.resize(gt[num],(224,224),interpolation=cv2.INTER_NEAREST)
        img = copy.copy(img_resize)
        gt = copy.copy(gt_resize)
        
        file_name = dir.split('/')[-1]
        for slice_num in range(len(img)):
            np.savez('../../data/data_npy_seg/'+file_name+'_'+str(slice_num)+'.npz',image = img[slice_num], label = gt[slice_num])
            with open('./lists/train_list.txt','a') as fp:
                fp.write(file_name+'_'+str(slice_num)+'\n')
                fp.close
            
        
        
        
        

    
