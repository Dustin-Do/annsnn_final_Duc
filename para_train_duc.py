import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import models
import argparse
import json
import modules
from utils import *
import numpy as np
import torch.nn.functional as F

model_name = 'vgg16'
device = 'cuda'
dataset = 'cifar10'
optimizer = 'sgd'


batch_size = 50
momentum = 0.9
decay = 1e-4
lr = 0.1

conf = [model_name,dataset]
save_name = '_'.join(conf) # 'save_name' = concatenation of all elements of 'conf'
log_dir = 'train_' + save_name
if not os.path.isdir(log_dir): # if 'log_dir' is not a directory
    os.makedirs(log_dir) # create path 'log_dir'

# Index 'ann_train_module' and 'snn_train_module' as modules which are already defined
ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()

# Define the model to be used is 'vgg16'
model = models.__dict__[model_name](num_classes=10, dropout=0)

# 'load_cv_data' is function defined in 'utils.py'
# datasets are defined: 'cifar10', 'cifar100', 'mnist', 'imagenet'
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=batch_size,
                 workers=0,
                 dataset=dataset,
                 data_target_dir=datapath[dataset]
                 )

loss_function1 = nn.CrossEntropyLoss()


# ---------------------- Define 'optimizer1' ---------------------------------------------------------------------------
if optimizer == 'sgd':
    optimizer1 = optim.SGD(ann_train_module.parameters(),
                               momentum=momentum,
                               lr=lr,
                               weight_decay=decay)
elif optimizer == 'adam':
    optimizer1 = optim.Adam(ann_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
elif optimizer == 'adamw':
    optimizer1 = optim.AdamW(ann_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
# ----------------------------------------------------------------------------------------------------------------------
writer = SummaryWriter(log_dir)

# 1.same as 'ann_train' function in 'para_train.py'
# 2.'ann_train' use variable 'train_dataloader' to train the model. The outputs below are returned. They are used to
#    evaluate the training result of ANN
# 3.outputs:
#       'ann_train_loss': loss between 'ann_outputs' and 'targets'
#       'ann_correct': nb of same elements between 'ann_predicted' and 'targets'
def ann_train(epoch):
    print('\n *****ann_train*****')
    global sum_k,cnt_k,train_batch_cnt
    net = model.to(device)

    print('\nEpoch: %d Para Train' % epoch)
    net.train()
    ann_train_loss = 0
    ann_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)): #tqdm is a library in Python which is used for creating Progress Meters or Progress Bars
        inputs, targets = inputs.to(device), targets.to(device)
        ann_outputs = net(inputs)
        ann_loss = loss_function1(ann_outputs, targets)
        ann_train_loss += (ann_loss.item()) # Sum up all 'ann_loss' patterns
        _, ann_predicted = ann_outputs.max(1) # find all cases along dim 1 (max in each row) of 'ann_outputs'

        tot = targets.size(0) # 'tot' equals to size of 1st dimension of 'targets'
        total += tot
        ac = ann_predicted.eq(targets).sum().item() # find elements of 'ann_predicted' that equal to 'target, then sum them up
        ann_correct += ac

        optimizer1.zero_grad()
        ann_loss.backward()
        # torch.nn.utils.clip_grad_norm_(ann_train_module.parameters(), 50)
        optimizer1.step() # All optimizers implement a step() method, that updates the parameters. It can be used as beside syntax
        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('encounter ann_loss', ann_loss)
            return False

        writer.add_scalar('Train/Acc', ac / tot, train_batch_cnt)
        writer.add_scalar('Train/Loss', ann_loss.item(), train_batch_cnt)
        train_batch_cnt += 1
    print('Para Train Epoch %d Loss:%.3f Acc:%.3f' % (epoch,
                                                      ann_train_loss,
                                                      ann_correct / total))
    writer.add_scalar('Train/EpochAcc', ann_correct / total, epoch)
    return


# 1.same as 'val' function in 'para_train.py'
# 2.'para_train_val' use variable 'test_dataloader' to test the model. It returns loss, accuracy which are the result of
#    the test
# 3.outputs:
#       'ann_test_loss': loss between 'ann_outputs' and 'targets'
#       'ann_correct': nb of same elements between 'ann_predicted' and 'targets'
#       'sum_k', 'cnt_k', 'last_k': WEIGHTs?
def para_train_val(epoch):
    print('\n *****para_train_val*****')
    global sum_k,cnt_k,test_batch_cnt,best_acc
    net = model.to(device)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            # '.register_forward_hook' registers a global forward hook for all the modules. It adds global state to the
            # nn.module module and it is only intended for debugging/profiling purposes.
            handles.append(m.register_forward_hook(modules.hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(device), targets.to(device)
            ann_outputs = net(inputs)
            ann_loss = loss_function1(ann_outputs, targets)

            if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
                print('encounter ann_loss', ann_loss)
                return False

            predict_outputs = ann_outputs.detach() # The 'detach()' method constructs a new view on a tensor which is
                                                   # declared not to need gradients, i.e., it is to be excluded from
                                                   # further tracking of operations, and therefore the subgraph involving
                                                   # this view is not recorded.
            ann_test_loss += (ann_loss.item())
            _, ann_predicted = predict_outputs.max(1) # '.max(1)' return max elements of rows in 'ann_predicted' and their positions

            tot = targets.size(0) # 'tot' = nb of rows in maxtrix 'targets'
            total += tot
            ac = ann_predicted.eq(targets).sum().item() # count nb of same elements between 'ann_predicted' and 'targets'
            ann_correct += ac

            # 'layerwise_k':greedy layer-wise pretraining that
            # allowed very deep neural networks to be successfully trained
            # 'layerwise_k' is defined above
            last_k = modules.layerwise_k(F.relu(ann_outputs), torch.max(ann_outputs))

            # The SummaryWriter class ('writer') is your main entry to log data for consumption and visualization by TensorBoard
            # Log 4 parameters of each loop (1 LOOP FOR 1 PICTURE ?) for later consumption and visualization
            # syntax: writer.add_scalar('',y,x)
            writer.add_scalar('Test/Acc', ac / tot, test_batch_cnt)
            writer.add_scalar('Test/Loss', ann_test_loss, test_batch_cnt)
            writer.add_scalar('Test/AvgK', (sum_k / cnt_k).item(), test_batch_cnt)
            writer.add_scalar('Test/LastK', last_k, test_batch_cnt)
            test_batch_cnt += 1

        print('Test Epoch %d Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (epoch,
                                                             ann_test_loss,
                                                             ann_correct / total,
                                                             sum_k / cnt_k, last_k))

    # Log 'ann_correct' for each epoch
    writer.add_scalar('Test/EpochAcc', ann_correct / total, epoch)

    # --------------------Save checkpoint-------------------------------------------------------------------------------
    # We just save the checkpoint when there is a better accuracy appears ('if acc > best_acc')
    # Parameters we save here are:
    #       'net.state_dict()': ???
    #       'acc': accuracy of the testing
    #       'epoch'
    acc = 100.*ann_correct/total  # 'acc' is percentage of correct elements over all elements
    if acc > best_acc:
        print('Saving checkpoint (para_train_val)...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir) # create directory 'log_dir'
        # save 'state' = ['net', 'acc', 'epoch'] to path train_vgg16_cifar10/vgg16_cifar10.pth
        # only have 1 file which saves the result
        torch.save(state, log_dir + '/%s.pth'%(save_name))
        best_acc = acc
    # ------------------------------------------------------------------------------------------------------------------

    # --------------------Schedule save checkpoint----------------------------------------------------------------------
    # We save checkpoint after every 10 epochs
    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    if (epoch + 1) % 10 == 0:
        print('Schedule Saving checkpoint (para_train_val)...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'avg_k': avg_k
        }
        # save 'state' = ['net', 'acc', 'epoch', 'avg_k'] to path train_vgg16_cifar10/vgg16_cifar10_pt_scheduled.pth
        # only have 1 file which saves the result
        torch.save(state, log_dir + '/%s_pt_scheduled.pth' % (save_name))
    for handle in handles:
        handle.remove()
    # ------------------------------------------------------------------------------------------------------------------




