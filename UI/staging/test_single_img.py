# import sys 
# sys.path.append("..") 

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from staging.network import *

from staging.crop import cropping
from staging.test_func import inference



def staging_single_img(single_img, image_Seg, test_time_adaptation):

    cropped_img = cropping(single_img, image_Seg)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/', help='list dir')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--max_epochs', type=int,
                        default=45, help='maximum epoch number to train')
    parser.add_argument('--root_path', type=str,
                        default='./data/final/iCTCF_6_24', help='root dir for data')
    parser.add_argument('--model_path', type=str,
                        default='./models_save/iCTCF_norm_bs4_5e5_de20/')
    parser.add_argument('--model_step', type=str,
                        default= 0)
    parser.add_argument('--max_step', type=str,
                        default= 30000)
    parser.add_argument('--batch_size', type=str,
                        default= 8)
    parser.add_argument('--base_lr', type=str,
                        default= 5e-5)
    parser.add_argument('--sequence_length', type=str,
                        default= 11)
    parser.add_argument('--train_txt', type=str,
                        default= 'iCTCF_train')
    parser.add_argument('--val_txt', type=str,
                        default= 'iCTCF_test')

    parser.add_argument('--num_classes', type=int,
                        default=5, help='output channel of network')
    args = parser.parse_args()


    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    model = resnet_lstm4_fle()
    
    model_path = '../staging/models_save/liver_fibrosis_staging/epoch_35.pth'
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    f0, f0_1, f0_2, f0_3, predicted_stage = inference(args, model, cropped_img, test_time_adaptation)
    
    return f0, f0_1, f0_2, f0_3, predicted_stage