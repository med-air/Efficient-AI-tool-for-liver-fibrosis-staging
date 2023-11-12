import numpy as np
import torch
from functools import reduce
from staging import tent

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

def inference(args, model, cropped_img, test_time_adaptation):
    if not test_time_adaptation:
        model.eval()
    else:
        model = setup_tent(model)
        model.reset()
        
    cropped_img = cropped_img[:,np.newaxis,:,:]
    image1 = cropped_img[:cropped_img.shape[0]-2]
    image2 = cropped_img[1:cropped_img.shape[0]-1]
    image3 = cropped_img[2:cropped_img.shape[0]]
    
    image_con = np.concatenate((image1, image2, image3), axis = 1)
    image_con = image_con[np.newaxis,:,:,:,:]
    cropped_img = torch.tensor(image_con)
    image_batch = cropped_img.cuda()
    if not test_time_adaptation:
        output = model(image_batch)
    else:
        output = model(image_batch,image_batch,mode=True)
    output = torch.mean(output,dim=0)
    output = output.view(1,4)
    output = torch.sigmoid(output)
    output = output.data.cpu().numpy()

    output_tran = np.float32(output)
    output_tran = tran_prediction(output_tran, np.array(0.5))
    output_tran = tran_class(output_tran) 

    pred_ = np.float32(output)


    return pred_[0,0], pred_[0,1], pred_[0,2], pred_[0,3], output_tran
        
        
