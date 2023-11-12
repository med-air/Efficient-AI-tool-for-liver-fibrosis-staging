# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F


def ordinal_regression_focal(predictions,targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target] = 1
    loss_sum = 0
    weight = torch.zeros_like(predictions)
    for class_index in range(modified_target.shape[1]):
        loss_sum  += nn.BCELoss(reduction='none')(predictions[:,class_index],modified_target[:,class_index])
        weight[:,class_index] = modified_target[:,class_index]*(1-predictions[:,class_index])**2+(1-modified_target[:,class_index])*predictions[:,class_index]**2
    weights_max,_=torch.max(weight,dim=1)
    return torch.mean(loss_sum)

def ordinal_regression(predictions,targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target+1] = 1

    return nn.MSELoss(reduction='mean')(predictions, modified_target)

def prediction2label(pred):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    
    pred_tran = (pred > 0.5).cumprod(axis=1)
    max_index = -1
    for j in range(pred_tran.shape[1]):
        if pred_tran[0][j] == 1:
            # max_index = j
            max_index +=  1
    
    return max_index

