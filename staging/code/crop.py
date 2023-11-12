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


def norm(im):
    im = (im - 0.5)*2
    return im

def Resize(img):
    originImg = np.zeros((img.shape[0],224,224)).astype(np.float32)
    for i in range(len(img)):
        Img = img[i]
        # Img = cv2.resize(Img, (512, 512), interpolation=cv2.INTER_NEAREST)
        Img = cv2.resize(Img, (224, 224))
        originImg[i] = Img
    return originImg

def cropping(single_img, image_Seg):
    LiverImg = copy.deepcopy(image_Seg)
    RawImg = copy.deepcopy(single_img)
    
    RawImg  = (RawImg + 1) / 2
    RawImg_ = copy.deepcopy(LiverImg)
    
    minD, maxD, max_index = catImg2(RawImg_)
    RawImg_[RawImg_ != 2] = 0
    RawImg_[RawImg_ == 2] = 1
    RawImg_ = Resize(RawImg_)
    mat = copy.deepcopy(RawImg_)
    mat[mat == 0] = 1      
    
    if RawImg.shape[0] != RawImg_.shape[0]:
        print(dir)
    RawImg[RawImg_==0] = RawImg[0,0,0]

    RawImg = RawImg * mat
 
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

        gt[ini] = RawImg_[j]
        ini += 1
        
    RawImg = copy.deepcopy(single_img)
    RawImg  = (RawImg + 1) / 2

    RawImg_2 = copy.deepcopy(LiverImg)
    minD, maxD, max_index = catImg1(RawImg_2)
    RawImg_2[RawImg_2 != 1] = 0
    RawImg_2[RawImg_2 == 1] = 1
    RawImg_2 = Resize(RawImg_2)
    mat = copy.deepcopy(RawImg_2)
    mat[mat == 0] = 1      

    RawImg[RawImg_2==0] = RawImg[0,0,0]
    RawImg = RawImg * mat
    ini = 0 

    for j in range(index,index+length):
        RawImg[j][result[ini]>0] = 0
        result[ini] = result[ini] + RawImg[j]
        # gt[ini] = gt[ini] + RawImg_[j]
        ini += 1
    result = norm(result)

    # resultImage = sitk.GetImageFromArray(result)     
    
    return result
