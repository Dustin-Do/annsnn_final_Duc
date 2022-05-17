import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import models
import json
import modules
import para_train_duc
import fast_train_duc
from utils import *
import numpy as np
import torch.nn.functional as F
import spikingjelly.clock_driven.functional as functional
import matplotlib.pyplot as plt
import spikingjelly.clock_driven.neuron as neuron

########################################################################################################################
#
# MODEL INITIALIZATION
#
########################################################################################################################
model_name = 'vgg16'
dataset = 'cifar10'
device = 'cuda'
optimizer = 'sgd'

momentum = 0.9
lr = 0.1    #leaning rate
schedule = [100, 150]
gammas = [0.1, 0.1]
decay = 1e-4
batch_size = 50
epoch = 200
acc_tolerance = 0.1
lam = 0.1
sharescale = True
scale_init = 2.5

best_acc = 0.0
start_epoch = 0
sum_k = 0.0
cnt_k = 0.0
last_k = 0
train_batch_cnt = 0
test_batch_cnt = 0

conf = [model_name,dataset]
save_name = '_'.join(conf) # 'save_name' = concatenation of all elements of 'conf'
log_dir = 'train_' + save_name
if not os.path.isdir(log_dir): # if 'log_dir' is not a directory
    os.makedirs(log_dir) # create path 'log_dir'
writer = SummaryWriter(log_dir)


# 'load_cv_data' is function defined in 'utils.py'
# datasets are defined: 'cifar10', 'cifar100', 'mnist', 'imagenet'
train_dataloader, test_dataloader = load_cv_data(data_aug=False,
                 batch_size=batch_size,
                 workers=0,
                 dataset=dataset,
                 data_target_dir=datapath[dataset]
                 )

# Define the model to be used is 'vgg16'
model = models.__dict__[model_name](num_classes=10, dropout=0)

# 'replace_maxpool2d_by_avgpool2d' & 'replace_relu_by_spikingnorm' are defined in 'modules.py'
model = modules.replace_maxpool2d_by_avgpool2d(model)
model = modules.replace_relu_by_spikingnorm(model,True)

for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)): # if 'm' is 'nn.Conv2d' (AND/OR)? 'nn.Linear'
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #Create weight explicitly by creating a random matrix based on Kaiming initialization
                                                                               #Read more:https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
        if hasattr(m,'bias') and m.bias is not None: # if 'm' has 'bias' and 'm.bias' is not none
            nn.init.zeros_(m.bias) # set value of 'm.bias' to 0
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, val=1)
        nn.init.zeros_(m.bias)

# --------------------- Define simulating configuration-----------------------------------------------------------------
model.to(device)
device = torch.device(device)
if device.type == 'cuda':
    print(f"=> cuda memory allocated: {torch.cuda.memory_allocated(device.index)}")
# ----------------------------------------------------------------------------------------------------------------------

# Index 'ann_train_module' and 'snn_train_module' as modules which are already defined
ann_train_module = nn.ModuleList()
snn_train_module = nn.ModuleList()

modules.divide_trainable_modules(model)

loss_function1 = nn.CrossEntropyLoss()
# Now, 'loss_function1' computes the cross entropy loss between input and target
loss_function2 = modules.new_loss_function
# Now, 'loss_function2' also computes loss between 'ann_out' & 'snn_out' by MSE or CosineSimilarity as defined above


# Optimizers are algorithms or methods used to minimize an error function
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

# --------------------------Define 'optimizer 2'----------------------------------------------------------------------------
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

########################################################################################################################
#
# SIMULATING FUNCTIONs
#
########################################################################################################################
def simulate(net, T, save_name, log_dir, ann_baseline=0.0):
    print('*****simulate*****')
    net.to(device) # link to device for simulation
    functional.reset_net(net)
    correct_t = {}

    # 'torch.no_grad(): Context-manager that disabled gradient calculation.
    # Disable gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
    # It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    with torch.no_grad():
        # 'net.eval()' is a kind of switch for some specific layers/parts of the model that behave differently during
        # training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn
        # off them during model evaluation, and '.eval()' will do it for you. In addition, the common practice for
        # evaluating/validation is using 'torch.no_grad()' in pair with 'model.eval()' to turn off gradients computation
        net.eval()
        correct = 0.0
        total = 0.0

        for batch, (img, label) in enumerate(test_dataloader):
            for t in range(T):
                out = net(img.to(device))
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                if t == 0:
                    out_spikes_counter = out
                else:
                    out_spikes_counter += out

                # 'keys()' method returns a view object. The view object contains the keys of the dictionary, as a list.
                if t not in correct_t.keys():
                    # 'out_spikes_counter.max(1)' return max element of each row of 'out_spikes_counter'
                    # 'float().sum().item()' sums up all float elements
                    # what is the meaning of variable 'correct_t'?
                    correct_t[t] = (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                else:
                    correct_t[t] += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            correct += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
            total += label.numel() # '.numel()' returns the total number of elements in the input tensor
            functional.reset_net(net)

            #--------------------------------Plotting-------------------------------------------------------------------
            fig = plt.figure()
            x = np.array(list(correct_t.keys())).astype(np.float32) + 1
            y = np.array(list(correct_t.values())).astype(np.float32) / total * 100
            plt.plot(x, y, label='SNN', c='b')
            if ann_baseline != 0:
                plt.plot(x, np.ones_like(x) * ann_baseline, label='ANN', c='g', linestyle=':')
                plt.text(0, ann_baseline + 1, "%.3f%%" % (ann_baseline), fontdict={'size': '8', 'color': 'g'})
            plt.title("%s Simulation \n[test samples:%.1f%%]" % (
                save_name, 100 * total / len(test_dataloader.dataset)
            ))
            plt.xlabel("T")
            plt.ylabel("Accuracy(%)")
            plt.legend()
            argmax = np.argmax(y)
            disp_bias = 0.3 * float(T) if x[argmax] / T > 0.7 else 0
            plt.text(x[argmax] - 0.8 - disp_bias, y[argmax] + 0.8, "MAX:%.3f%% T=%d" % (y[argmax], x[argmax]),
                     fontdict={'size': '12', 'color': 'r'})

            plt.scatter([x[argmax]], [y[argmax]], c='r')
            print('[SNN Simulating... %.2f%%] Acc:%.3f' % (100 * total / len(test_dataloader.dataset),
                                                                     correct / total))
            acc_list = np.array(list(correct_t.values())).astype(np.float32) / total * 100
            np.save(log_dir + '/snn_acc-list' + ('-constant'), acc_list)
            plt.savefig(log_dir + '/sim_' + save_name + ".jpg", dpi=1080)

            from PIL import Image
            im = Image.open(log_dir + '/sim_' + save_name + ".jpg")
            totensor = transforms.ToTensor()
            plt.close()
            # ----------------------------------------------------------------------------------------------------------
        acc = correct / total
        print('SNN Simulating Accuracy:%.3f' % (acc ))


def simulate_by_filename(save_name):
    print('\n\n\n########################################################################################################')
    print('Start simulate by filename')
    print('########################################################################################################')

    print('Filename: %s' %save_name)
    model = models.__dict__[model_name](num_classes=10, dropout=0)
    model = modules.replace_maxpool2d_by_avgpool2d(model) #function from 'modules.py'
    model = modules.replace_relu_by_spikingnorm(model,True) #function from 'modules.py'
    state_dict = torch.load('train_vgg16_cifar10/%s.pth' % save_name)
    # In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the
    # model’s parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps
    # each layer to its parameter tensor.
    ann_acc = state_dict['acc']
    model.load_state_dict(state_dict['net'])
    model = modules.replace_spikingnorm_by_ifnode(model)
    simulate(model.to(device), T=100, save_name='%s' % save_name, log_dir=log_dir, ann_baseline=ann_acc)


########################################################################################################################
#
# PHASE 1: TRAINING FOR WEIGHTS
#
########################################################################################################################
print('\n\n\n########################################################################################################')
print('Start Phase 1: train for weights')
print('########################################################################################################')

for epoch in range(start_epoch, start_epoch + epoch):
    print('\n*********************************************')
    print('Epoch: ', epoch)
    print('*********************************************')

    modules.adjust_learning_rate(optimizer1, epoch)

    if epoch==start_epoch:
        para_train_duc.val(epoch)
    ret = para_train_duc.ann_train(epoch)
    if ret == False:
        break
    # output of 'para_train_val': 'epoch', 'ann_test_loss', 'ann_correct', 'sum_k', 'cnt_k', 'last_k'
    para_train_duc.val(epoch)
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data)


########################################################################################################################
#
# PHASE 2: TRAINING FOR FAST INFERENCE
#
########################################################################################################################
dataset = train_dataloader.dataset

# divide 'dataset' (50000 images) into 'train_set' (40000 images) and 'val_set' (10000 images)
train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])

# load data from 'train_set' and 'val_set' and save to 'train_dataloader' and 'val_data_loader'
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# In PyTorch, the learnable parameters (i.e. weights and biases) of torch.nn.Module model are contained in the model’s
# parameters (accessed with model.parameters()). A state_dict is simply a Python dictionary object that maps each layer
# to its parameter tensor.
# Load weights from file having path 'train_vgg16_cifar10/vgg16_cifar10.pth', which contains:
#       'net.state_dict()': ???
#       'acc': accuracy of the testing
#       'epoch'
model.load_state_dict(torch.load('train_vgg16_cifar10/vgg16_cifar10.pth')['net'])



# ----------------------------  Scaling--------------------------------------------------------------------------
# Feature scaling in machine learning is one of the most critical steps during the pre-processing of data before
# creating a machine learning model.
# Use scaling to normalize all features of model for better evaluating the loss
if sharescale:
    first_scale = None
    sharescale = nn.Parameter(torch.Tensor([scale_init]))
    for m in model.modules():
        if isinstance(m, modules.SpikingNorm):
            setattr(m, 'scale', sharescale) # set the 'scale' of 'm' equals to 'sharescale'
            m.lock_max = True
# ----------------------------------------------------------------------------------------------------------------------

modules.divide_trainable_modules(model)

# define opt2
lr = 0.001
inspect_interval = 100

best_acc = fast_train_duc.get_acc(val_dataloader)

for e in range(0, epoch): # 'epoch'=200 as defined in line 35
    print("Epoch: ",e)
    modules.adjust_learning_rate(optimizer2, e)
    ret = fast_train_duc.snn_train(e)
    if ret == False:
        break
    print("\nThres:")
    for n, m in model.named_modules():
        if isinstance(m, modules.SpikingNorm):
            print('thres', m.calc_v_th().data, 'scale', m.calc_scale().data, 'scale_t',m.scale.data)
            # '.calc_v_th()' and '.calc_scale()' are 2 functions defined in 'modules.py'


########################################################################################################################
#
# SIMULATION
#
########################################################################################################################
simulate_by_filename('vgg16_cifar10_ft_scheduled')
simulate_by_filename('vgg16_cifar10_pt_scheduled')






























































































