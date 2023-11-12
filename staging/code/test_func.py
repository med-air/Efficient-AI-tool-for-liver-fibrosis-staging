import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.dataset import Synapse_dataset, RandomGenerator
from tqdm import tqdm
from sklearn import metrics
from loss import prediction2label
import torch.nn.functional as F
from functools import reduce
from tools import *
from eval import *
import tent


def tran_prediction(pred,thr):
    pred = pred > thr
    pred_ = pred.astype(int)
    return pred_

def tran_class(pred_sample):
    sample_class = []
    for k in range(len(pred_sample)):
        list_ = pred_sample[:k+1]
        ln = reduce(lambda x,y:x*y,list_)
        sample_class.append(ln*np.sum(list_))
    return np.max(sample_class) 

# def Acc_AUC(predict, gt,class_num):
#     predict = np.array(predict)
#     predict_ = predict[:,class_num]
#     gt = np.array(gt)
#     gt_ = np.zeros(len(gt)).astype(np.float32)
#     gt_[gt<=class_num] = 0
#     gt_[gt>class_num] = 1
#     auc, auc_cov = delong_roc_variance(gt_,predict_)
    
#     print('{} AUC:'.format(class_num),auc, np.sqrt(auc_cov), auc-1.96*np.sqrt(auc_cov), auc+1.96*np.sqrt(auc_cov))
#     pred_half = tran_prediction(predict, np.array(0.5))
#     predicted_class = []
#     for sample_index in range(len(predict)):
#         sample_class = tran_class(pred_half[sample_index]) 
#         if sample_class <= class_num:
#             predicted_class.append(0)
#         else:
#             predicted_class.append(1)
#     confusion = metrics.confusion_matrix(gt_, predicted_class,labels=[0,1,2,3,4])
#     specificity_ = 0 
#     if float(confusion[0,0]+confusion[1,0])!=0:
#         specificity_ = float(confusion[0,0])/float(confusion[0,0]+confusion[1,0])
#     specificity = 0 
#     if float(confusion[0,0]+confusion[0,1])!=0:
#         specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
#     sensitivity = 0
#     if float(confusion[1,1]+confusion[1,0])!=0:
#         sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
#     precision = 0
#     if float(confusion[1,1]+confusion[0,1])!=0:
#         precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
#     accuracy = 0
#     if np.sum(confusion) != 0:
#         accuracy = (float(confusion[1,1]) + float(confusion[0,0])) / np.sum(confusion)
#     #print(metrics.classification_report(gt_,pred))
#     #print(metrics.confusion_matrix(gt_,pred))
#     specificity_std = np.sqrt(specificity*(1-specificity)/float(confusion[0,0]+confusion[0,1]))
#     # specificity_std_ = np.sqrt(specificity_*(1-specificity_)/float(confusion[0,0]+confusion[1,0]))
#     sensitivity_std = np.sqrt(sensitivity*(1-sensitivity)/float(confusion[1,1]+confusion[1,0]))
#     #precision_std = np.sqrt(precision*(1-precision)/float(confusion[1,1]+confusion[0,1]))
#     accuracy_std = np.sqrt(accuracy*(1-accuracy)/np.sum(confusion))
#     print('Accuracy:',accuracy, int(float(confusion[0,0])+float(confusion[1,1])), '/', int(np.sum(confusion)), accuracy-1.96*accuracy_std, accuracy+1.96*accuracy_std)
#     print('Specificity:',specificity, int(float(confusion[0,0])), '/', int(float(confusion[0,0]+confusion[0,1])), specificity-1.96*specificity_std, specificity+1.96*specificity_std)
#     # print('Specificity_:',specificity_, int(float(confusion[0,0])), '/', int(float(confusion[0,0]+confusion[1,0])))
#     print('Sensitivity:',sensitivity, int(float(confusion[1,1])), '/', int(float(confusion[1,1]+confusion[1,0])),  sensitivity-1.96*sensitivity_std, sensitivity+1.96*sensitivity_std)
#     #print('Precision:',precision, int(float(confusion[1,1])), '/', int(float(confusion[1,1]+confusion[0,1])),  precision-1.96*precision_std, precision+1.96*precision_std)
#     print("============================================")
#     return auc

def setup_tent(model):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    tent_model = tent.Tent(model, optimizer=torch.optim.SGD(params, lr = 1e-4))
    # print("model for adaptation: %s", model)
    # print("params for adaptation: %s", param_names)
    return tent_model

def inference(args, model, error_name, test_save_path=None, test_time=False):
    db_test = Synapse_dataset(base_dir='../../data/data_npy_cls', list_dir=args.list_dir, split=args.val_txt,is_train = False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    if not test_time:
        model.eval()
    else:
        model = setup_tent(model)
        model.reset()

        
    result = []
    result_ = []
    Y_val_set = []
    iteration_error_name = []
    prediction_epoch = []
    y_train =  []
    case_name_list = []
    # lr1,lr2,lr3,lr4 = train_the_regression()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image_batch, label_batch, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name']
        case_name_list.append(case_name)
        image_batch = image_batch.cuda()
        if not test_time:
            output = model(image_batch)
        else:
            output = model(image_batch,image_batch,mode=True)
        output = torch.mean(output,dim=0)
        output = output.view(1,4)
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
        prediction_epoch.append(output)
        # output = prediction_calibration(output,lr1,lr2,lr3,lr4)
        output_tran = np.float32(output)
        output_tran = tran_prediction(output_tran, np.array(0.5))
        output_tran = tran_class(output_tran) 
        # output_tran = prediction2label(output)
        # _, preds_phase = torch.max(output.data, 1)
        
        pred_ = np.float32(output)
        label_batch = np.float32(label_batch.data.cpu().numpy())
        y_train.append(label_batch[0])
        # pred_result = np.float32(output_tran.data.cpu().numpy())
        if output_tran != label_batch[0]:
            error_name[case_name[0]][output_tran] += 1
            error_name[case_name[0]][int(label_batch[0])] = -1
            iteration_error_name.append(case_name)


        result_.append(pred_[0])
        result.append(output_tran)
        Y_val_set.append(label_batch[0])
    
    print(metrics.classification_report(Y_val_set,result))
    print(metrics.confusion_matrix(Y_val_set,result,labels=[0,1,2,3,4]))
    # print(iteration_error_name)
        #print(Y_val_set)
        #print(result_)
    
    return error_name,np.array(result_),np.array(Y_val_set),case_name_list
        
