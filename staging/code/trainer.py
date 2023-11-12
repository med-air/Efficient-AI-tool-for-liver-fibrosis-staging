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
from  loss import ordinal_regression_focal
import torch.nn.functional as F
from tools import get_error_name


def trainer_synapse(args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split=args.train_txt,is_train = True,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train3")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    class PadSequence:
        def __call__(self,batch):
            sorted_batch = sorted(batch,key=lambda x: x['image'].shape[0],reverse=True)
            sequences = [torch.from_numpy(x['image']) for x in sorted_batch]
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True)

            labels = [x['label'] for x in sorted_batch]
            labels = torch.tensor(np.array(labels))
            return sequences_padded, labels

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, collate_fn=PadSequence())
    device = torch.device("cuda:0")

    model = torch.nn.DataParallel(model) 
    model = model.to(device)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.module.parameters(), lr = args.base_lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma=0.5)  
    error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs 
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(args.model_step,max_epoch+1), ncols=70)
    for epoch_num in iterator:
        error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            output = model(image_batch)
            output = output.view(-1,image_batch.shape[1],4)
            output = torch.mean(output,dim=1)

            output = output.view(output.shape[0],4)
            output = torch.sigmoid(output)

            loss = ordinal_regression_focal(output, label_batch)
            optimizer.zero_grad()
            loss.backward()
            scheduler.step(epoch_num)
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            

        
        save_interval = 50  

        writer.add_scalar('info/total_loss', loss, iter_num)
        logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        error_name,result_,Y_val_set,case_name_list = inference(args, model,error_name)

        model.train()
        print(args.model_path)
        
        if epoch_num % 1 ==0:
            print(error_name)
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            save_result_path = os.path.join(snapshot_path, 'result_' + str(epoch_num) + '.npy')
            np.save(save_result_path,{'result':result_,'label':Y_val_set, 'name':case_name_list})
            torch.save(model.module.state_dict(), save_mode_path)
            # logging.info("save model to {}".format(save_mode_path))

            

    writer.close()

    return "Training Finished!"
