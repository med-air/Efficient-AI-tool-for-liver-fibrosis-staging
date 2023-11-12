# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:00:59 2022

@author: wama
"""

import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from datasets.dataset import Synapse_dataset, RandomGenerator
from test_func import inference
import torch.nn.functional as F
from tools import get_error_name
import time
from network import *


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
                    default=35, help='maximum epoch number to train')
parser.add_argument('--root_path', type=str,
                    default='../../data/data_npy_cls', help='root dir for data')
parser.add_argument('--model_path', type=str,
                    default='../models_save/liver_fibrosis_staging/')
parser.add_argument('--model_step', type=str,
                    default= 35)
parser.add_argument('--max_step', type=str,
                    default= 30000)
parser.add_argument('--batch_size', type=str,
                    default= 1)
parser.add_argument('--base_lr', type=str,
                    default= 5e-5)
parser.add_argument('--sequence_length', type=str,
                    default= 11)
parser.add_argument('--train_txt', type=str,
                    default= 'train_list')
parser.add_argument('--val_txt', type=str,
                    default= 'test_list')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--test_time', type=bool,
                    default=True)
args = parser.parse_args()


model = resnet_lstm4_fle()

model_path = args.model_path  +'epoch_'+ str(args.model_step) + '.pth'
model.load_state_dict(torch.load(model_path))
model = model.cuda()


error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')

error_name,result_,Y_val_set,case_name_list = inference(args, model,error_name,args.model_step,test_time = args.test_time)
