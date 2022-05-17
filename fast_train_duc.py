import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
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
lam = 0.1
inspect_interval = 100
acc_tolerance = 0.1
epoch = 200


conf = [model_name,dataset]
save_name = '_'.join(conf) # 'save_name' = concatenation of all elements of 'conf'
log_dir = 'train_' + save_name
if not os.path.isdir(log_dir): # if 'log_dir' is not a directory
    os.makedirs(log_dir) # create path 'log_dir'

writer = SummaryWriter(log_dir)


# Define the model to be used is 'vgg16'
model = models.__dict__[model_name](num_classes=10, dropout=0)


# Index 'ann_train_module' and 'snn_train_module' as modules which are already defined
ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()


# 'load_cv_data' is function defined in 'utils.py'
# datasets are defined: 'cifar10', 'cifar100', 'mnist', 'imagenet'
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=batch_size,
                 workers=0,
                 dataset=dataset,
                 data_target_dir=datapath[dataset]
                 )


def new_loss_function(ann_out, snn_out, k, func='cos'):
    print('new_loss_function')
    if func == 'mse':
        f = nn.MSELoss()
        diff_loss = f(ann_out, snn_out) # assign 'diff_loss' equal to MSE between 'ann_out' and 'snn_out'
    elif func == 'cos':
        f = nn.CosineSimilarity(dim=1, eps=1e-6) # read more about CosineSimilarity func: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
        diff_loss = 1.0 - torch.mean(f(ann_out, snn_out))
    else:
        assert False
    loss = diff_loss + lam * k
    return loss, diff_loss

loss_function1 = nn.CrossEntropyLoss()
loss_function2 = new_loss_function()

dataset = train_dataloader.dataset
# divide 'dataset' (50000 images) into 'train_set' (40000 images) and 'val_set' (10000 images)
train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])

# load data from 'train_set' and 'val_set' and save to 'train_dataloader' and 'val_data_loader'
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --------------------------Define optimizer----------------------------------------------------------------------------
if optimizer == 'sgd':
    optimizer2 = optim.SGD(snn_train_module.parameters(),
                           momentum=momentum,
                           lr=lr,
                           weight_decay=decay)
elif optimizer == 'adam':
    optimizer2 = optim.Adam(snn_train_module.parameters(),
                           lr=lr,
                           weight_decay=decay)
# ----------------------------------------------------------------------------------------------------------------------


# 1.same as 'snn_train' function in 'fast_train.py'
# 2.'snn_train' uses 'train_dataloader' to train the model, it trains ANN and SNN, then compare results
# 3.outputs:
#       'snn_dist_loss', 'snn_fast_loss':
#                Two losses are considered ('fast_loss' and 'dist_lost'), which are loss between ANN & SNN training outputs
#                'snn_dist_loss' is cumulation of 'dist_loss'
#       'snn_correct': nb of same elements between 'snn_predicted' and 'targets' of SNN (not output of ANN training)
def snn_train(epoch):
    print('\n *****snn_train*****')
    global sum_k, cnt_k, train_batch_cnt, last_k
    net = model.to(device)

    print('\nEpoch: %d Fast Train' % epoch)
    net.train()
    snn_fast_loss = 0
    snn_dist_loss = 0
    snn_correct = 0
    total = 0

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(modules.hook))

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader)):
        sum_k = 0
        cnt_k = 0
        # ----------------------Run ANN training------------------------------------------------------------------------
        inputs, targets = inputs.to(device), targets.to(device)
        ann_outputs = net(inputs)
        ann_loss = loss_function1(ann_outputs, targets)

        if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
            print('encounter ann_loss', ann_loss)
            return False

        # 'detach()' method constructs a new view on a tensor which is declared not to need gradients, i.e., it is
        # to be excluded from further tracking of operations, and therefore the subgraph involving this view is not recorded.
        predict_outputs = ann_outputs.detach()
        _, ann_predicted = predict_outputs.max(1)
        # --------------------------------------------------------------------------------------------------------------

        # ----------------------Run SNN training------------------------------------------------------------------------
        snn_outputs = net(inputs)

        # 'F.relu(snn_outputs)' returns positive elements, others are set to 0
        # 'torch.mac(snn_outputs)' return max element of 'snn_outputs'
        # 'layerwise_k' is defined above
        last_k = modules.layerwise_k(F.relu(snn_outputs), torch.max(snn_outputs))

        # 'predict_outputs' is output of ANN (i.e. 'ann_outputs'), 'snn_outputs' is output of SNN
        # 'loss_function2' uses MSE or CosineSimilarity technique to calculates difference between 'predict_outputs'
        #  (returned by ANN) and 'snn_outputs'.
        # fast_loss = dist_loss + lam * [(sum_k + last_k) / (cnt_k + 1)]
        fast_loss, dist_loss = loss_function2(predict_outputs, snn_outputs, (sum_k + last_k) / (cnt_k + 1))

        snn_dist_loss += dist_loss.item()
        snn_fast_loss += fast_loss.item()
        optimizer2.zero_grad()
        fast_loss.backward()
        optimizer2.step()

        _, snn_predicted = snn_outputs.max(1)
        tot = targets.size(0)
        total += tot
        sc = snn_predicted.eq(targets).sum().item()
        snn_correct += sc
        # --------------------------------------------------------------------------------------------------------------

        # The SummaryWriter class ('writer') is your main entry to log data for consumption and visualization by TensorBoard
        # Log 4 parameters of each loop (1 LOOP FOR 1 PICTURE ?) for later consumption and visualization
        # syntax: writer.add_scalar('',y,x)
        writer.add_scalar('Train/Acc', sc / tot, train_batch_cnt)
        writer.add_scalar('Train/DistLoss', dist_loss, train_batch_cnt)
        writer.add_scalar('Train/AvgK', (sum_k / cnt_k).item(), train_batch_cnt)
        writer.add_scalar('Train/LastK', last_k, train_batch_cnt)
        train_batch_cnt += 1

        # 'inspect_interval' is a time interval which is used to observe the data progress
        if train_batch_cnt % inspect_interval == 0:
            if not snn_val(train_batch_cnt):
                return False
            net.train()
    print('Fast Train Epoch %d Loss:%.3f Acc:%.3f' % (epoch,
                                                      snn_dist_loss,
                                                      snn_correct / total))

    writer.add_scalar('Train/EpochAcc', snn_correct / total, epoch)
    for handle in handles:
        handle.remove()
    return True


# 1.same as 'get_acc' function in 'fast_train.py'
# 2.output:
#       'snn_acc': nb of same elements between 'predicted' (i.e. testing output of 'model' on 'val_dataloader') and 'targets'
# Used to update the best accuracy to save the checkpoint in 'snn_val'.
# Why in 'para_train_val.py', the update is contained in the file, not in separate file like in SNN case???
def get_acc(val_dataloader):
    print('\n *****get_acc*****')
    global model
    net = model
    net.to(device)

    net.eval()
    correct = 0
    total = 0
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            m.lock_max = True
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    snn_acc = correct / total
    return snn_acc


# 1.same as 'val' function in 'fast_train.py'
# 2.same as 'para_train_val' defined above
# 3.used for test dataset
# 3.why 'snn_val' uses ANN training instead of SNN???
def snn_val(iter):
    print('\n *****snn_val*****')
    global sum_k, cnt_k, test_batch_cnt, best_acc, last_k, best_avg_k
    net = model.to(device)

    handles = []
    for m in net.modules():
        if isinstance(m, modules.SpikingNorm):
            handles.append(m.register_forward_hook(modules.hook))

    net.eval()
    ann_test_loss = 0
    ann_correct = 0
    total = 0
    with torch.no_grad(): #disable gradient calculation.
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_dataloader)):
            sum_k = 0
            cnt_k = 0
            inputs, targets = inputs.to(device), targets.to(device)
            ann_outputs = net(inputs)
            ann_loss = loss_function1(ann_outputs, targets)

            if np.isnan(ann_loss.item()) or np.isinf(ann_loss.item()):
                print('encounter ann_loss', ann_loss)
                return False

            predict_outputs = ann_outputs.detach()
            ann_test_loss += (ann_loss.item())
            _, ann_predicted = predict_outputs.max(1)

            tot = targets.size(0)
            total += tot
            ac = ann_predicted.eq(targets).sum().item()
            ann_correct += ac

            last_k = modules.layerwise_k(F.relu(ann_outputs), torch.max(ann_outputs))
            # SummaryWriter class ('writer') is a main entry to log data for consumption, visualization by TensorBoard
            writer.add_scalar('Test/Acc', ac / tot, test_batch_cnt)
            writer.add_scalar('Test/Loss', ann_test_loss, test_batch_cnt)
            writer.add_scalar('Test/AvgK', (sum_k / cnt_k).item(), test_batch_cnt)
            writer.add_scalar('Test/LastK', last_k, test_batch_cnt)
            test_batch_cnt += 1
            #–––––-----------–––––-----------–––––-----------–––––-----------–––––-----------–––––-----------
        print('Test Iter %d Loss:%.3f Acc:%.3f AvgK:%.3f LastK:%.3f' % (iter,
                                                                         ann_test_loss,
                                                                         ann_correct / total,
                                                                         sum_k / cnt_k, last_k))
    writer.add_scalar('Test/IterAcc', ann_correct / total, iter)

    # Save checkpoint.
    avg_k = ((sum_k + last_k) / (cnt_k + 1)).item()
    acc = 100. * ann_correct / total
    if acc < (best_acc - acc_tolerance)*100.:
        return False
    if acc > (best_acc - acc_tolerance)*100. and best_avg_k > avg_k:
        test_acc = get_acc(test_dataloader)
        print('Saving checkpoint (snn_val)...')
        state = {
            'net': net.state_dict(),
            'acc': test_acc * 100,
            'epoch': epoch,
            'avg_k': avg_k
        }
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        torch.save(state, log_dir + '/%s_[%.3f_%.3f_%.3f].pth' % (save_name,
                                                                       lam,test_acc * 100,
                                                                       ((sum_k + last_k) / (cnt_k + 1)).item() ))
        best_avg_k = avg_k

    if (epoch + 1) % 10 == 0:
        print('Schedule saving checkpoint (snn_val)...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, log_dir + '/%s_ft_scheduled.pth' % (save_name))
    for handle in handles:
        handle.remove()
    return True





































