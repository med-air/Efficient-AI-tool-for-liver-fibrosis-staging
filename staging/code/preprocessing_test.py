
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:21:58 2021
@author: wenaoma
"""

import numpy as np
import os
import random
import SimpleITK as sitk
import cv2
#from skimage import transform
import h5py
import copy

def norm(im):
    mean_ = np.mean(im)
    std_ = np.std(im)
    im = (im - mean_) / std_
    max_ = np.max(im)
    min_ = np.min(im)
    im  = (im - min_) / (max_ -  min_)
    im = (im - 0.5)*2
    return im

def catImg(img):
    D,H,W = np.where(img!=0)
    minD = np.min(D)
    maxD = np.max(D)
    num =  0
    index = 0
    for i in range(len(img)):
        single = img[i,:,:]
        a,b = np.where(single>=1)
        num_ = len(a)
        if num_ >= num:
            num = num_
            index = i
    return minD,maxD,index


def catImg1(img):
    D,H,W = np.where(img==1)
    minD = np.min(D)
    maxD = np.max(D)
    num =  0
    index = 0
    for i in range(len(img)):
        single = img[i,:,:]
        a,b = np.where(single==1)
        num_ = len(a)
        if num_ >= num:
            num = num_
            index = i
    return minD,maxD,index

def catImg2(img):
    D,H,W = np.where(img==2)
    if len(D)!=0:
        minD = np.min(D)
        maxD = np.max(D)
        num =  0
        index = 0
        for i in range(len(img)):
            single = img[i,:,:]
            a,b = np.where(single==2)
            num_ = len(a)
            if num_ >= num:
                num = num_
                index = i
    else:
        index = 15
        minD = 0
        maxD = 0

    return minD,maxD,index    


def Resize(img):
    originImg = np.zeros((img.shape[0],224,224)).astype(np.float32)
    for i in range(len(img)):
        Img = img[i]
        # Img = cv2.resize(Img, (512, 512), interpolation=cv2.INTER_NEAREST)
        Img = cv2.resize(Img, (224, 224))
        originImg[i] = Img
    return originImg


train_file_path = '../../data/test/' 


for dir, file, images in os.walk(train_file_path):
    if 'img_raw.nii' in images:
        Liver = sitk.ReadImage(os.path.join(dir, 'prediction.nii.gz'))
        LiverImg = sitk.GetArrayFromImage(Liver)
        LiverImg = LiverImg.astype(np.uint8)
        
        # flag = dir.split('/')[-1] in list_
        
        Raw = sitk.ReadImage(os.path.join(dir, 'img_raw.nii.gz'))
        RawImg = sitk.GetArrayFromImage(Raw)
        # RawImg = norm(RawImg)
        # if  dir.split('/')[-1] in list_:
        
        RawImg  = (RawImg + 1) / 2
        RawImg_ = copy.deepcopy(LiverImg)
        # if dir.split('/')[-1]== 'R06921301 2min':
        #     print(1)
        minD, maxD, max_index = catImg2(RawImg_)
        RawImg_[RawImg_ != 2] = 0
        RawImg_[RawImg_ == 2] = 1
        RawImg_ = Resize(RawImg_)
        mat = copy.deepcopy(RawImg_)
        mat[mat == 0] = 1      
        
        if RawImg.shape[0] != RawImg_.shape[0]:
            print(dir)
        RawImg[RawImg_==0] = RawImg[0,0,0]
        # RawImg = Resize(RawImg)
        RawImg = RawImg * mat

        #length = maxD - minD + 1
        # length = 13
        # length = RawImg.shape[0]
        if RawImg.shape[0] >=30:
            length = 30
            if (max_index+length//2)<= len(RawImg) and (max_index - length//2-1)>=0:
                index = max_index - length//2-1
            elif (max_index+length//2)>len(RawImg):
                index = len(RawImg) - length
            else:
                index = 0
        else:
            length = RawImg.shape[0]
            index = 0

        result = np.zeros((length,RawImg.shape[1],RawImg.shape[2])).astype(np.float32)
        gt = np.zeros((length,RawImg.shape[1],RawImg.shape[2])).astype(np.float32)

        
        ini = 0 

        for j in range(index,index+length):
            #im = RawImg[j]
            
            result[ini] = RawImg[j]
            gt[ini] = RawImg_[j]
            ini += 1

        Raw = sitk.ReadImage(os.path.join(dir, 'img_raw.nii.gz'))
        RawImg = sitk.GetArrayFromImage(Raw)
        #RawImg = norm(RawImg)
        # if  dir.split('/')[-1] in list_:
        
        RawImg  = (RawImg + 1) / 2

        RawImg_2 = copy.deepcopy(LiverImg)
        minD, maxD, max_index = catImg1(RawImg_2)
        RawImg_2[RawImg_2 != 1] = 0
        RawImg_2[RawImg_2 == 1] = 1
        RawImg_2 = Resize(RawImg_2)
        mat = copy.deepcopy(RawImg_2)
        mat[mat == 0] = 1      
        
        # if RawImg.shape[0] != RawImg_.shape[0]:
        #     print(dir)
        RawImg[RawImg_2==0] = RawImg[0,0,0]
        # RawImg = Resize(RawImg)
        RawImg = RawImg * mat
        # RawImg = Resize(RawImg)
        # RawImg[RawImg_>0] = 0

        
        #length = maxD - minD + 1
        # length = 13
    
        ini = 0 


        for j in range(index,index+length):
            RawImg[j][result[ini]>0] = 0
            result[ini] = result[ini] + RawImg[j]
            # gt[ini] = gt[ini] + RawImg_[j]
            ini += 1

        # result = Resize(result)
        #result = norm(result)
        #result = Resize_3D(result,gt)
        #result_ = Resize(result)
        resultImage = sitk.GetImageFromArray(result)     
        fullname = os.path.join(dir, './convert_img_fle.nii.gz')
        sitk.WriteImage(resultImage,fullname)
        
        saved_name = dir.split('/')[-1] 
        
        f = h5py.File('../../data/data_npy_cls/'  + saved_name+'.npy.h5','w') 
        f['image'] = result                
        if  dir.split('/')[-2][-1] == 'n':
            print(dir)
        f['label'] = int(dir.split('/')[-2][-1])   
        # f['pathogeny'] = int(dir.split('/')[-2][-1])      
        f.close()  
        with open('./lists/test_list.txt','a') as fp:
            fp.write(saved_name+'\n')
            fp.close

        