import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import MetaDataLoader
from model.base_model import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm
import argparse
import warnings
import pdb
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser(description="MPN")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--loss_fra_reconstruct', type=float, default=1.00, help='weight of the frame reconstruction loss')
parser.add_argument('--loss_fea_reconstruct', type=float, default=1.00, help='weight of the feature reconstruction loss')
parser.add_argument('--loss_distinguish', type=float, default=0.0001, help='weight of the feature distinction loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=128, help='channel dimension of the features')
parser.add_argument('--pdim', type=int, default=128, help='channel dimension of the prototypes')
parser.add_argument('--psize', type=int, default=10, help='number of the prototypes')
parser.add_argument('--alpha', type=float, default=0.5, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='.data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--resume', type=str, default='exp/pretrain_model.pth', help='file path of resume pth')
parser.add_argument('--debug', type=bool, default=False, help='if debug')
# meta setting
parser.add_argument('--task_size', type=int, default=4, help='task size for meta training, 1 for meta training, the others for meta validation')
parser.add_argument('--segs', type=int, default=32, help='number of segs when dividing the videos for meta training')
parser.add_argument('--meta_base', type=str, default='exp/pretrain_model.pth', help='file path of resume pth')
parser.add_argument('--meta_init_lr', type=float, default=1e-6, help='meta initial learning rate')
parser.add_argument('--meta_alpha_init', type=float, default=1e-6, help='meta initial alpha')
parser.add_argument('--meta_alpha_lr', type=float, default=1e-6, help='meta initial alpha learning rate')
parser.add_argument('--meta_epoch', type=int, default=1000, help='meta training epoch')
args = parser.parse_args()

torch.manual_seed(2021)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+args.dataset_type+"/training/frames"

# Loading dataset
train_dataset = MetaDataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, task_size=args.task_size, segs=args.segs)

train_size = len(train_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)


# Model setting
model = convAE(args.c, args.t_length, args.psize, args.fdim, args.pdim)
model.cuda()


if len(args.gpus[0])>1:
  model = nn.DataParallel(model)

start_epoch = 0

if os.path.exists(args.meta_base):
  ckpt = args.meta_base


if ckpt != '':
  print('Load AE parameters from:', ckpt)
  saved_state_dict = torch.load(ckpt)['state_dict'].state_dict()
  model.load_state_dict(saved_state_dict, strict=True)

model.set_learnable_params(['memory','ohead'])


## Initialize Meta Params(learning rate) Net##
if os.path.exists(args.resume):
  print('Resume model from '+ args.resume)
  ckpt = args.resume
  checkpoint = torch.load(ckpt)
  start_epoch = checkpoint['epoch']
  meta_init = checkpoint['meta_init']
  meta_alpha = checkpoint['meta_alpha']
else:
  meta_init = OrderedDict()
  for k,p in model.get_learnable_params().items():
    meta_init[k] = Variable(p.data.clone(), requires_grad=True)

  meta_alpha = OrderedDict()
  for k,p in model.get_learnable_params().items():
    alpha = Variable(p.data.clone(), requires_grad=True)
    alpha.data.fill_(args.meta_alpha_init)
    meta_alpha[k] = alpha

meta_init_params = [p for k,p in meta_init.items()]
names = [k for k,p in meta_init.items()]
meta_init_optimizer = optim.Adam(meta_init_params, lr = args.meta_init_lr)

## Initialize Meta Alpha(learning rate) Net##
meta_alpha_params = [p for k,p in meta_alpha.items()]
meta_alpha_optimizer = optim.Adam(meta_alpha_params, lr = args.meta_alpha_lr)


if os.path.exists(args.resume):
  meta_init_optimizer.load_state_dict(checkpoint['meta_init_optimizer'])
  meta_alpha_optimizer.load_state_dict(checkpoint['meta_alpha_optimizer'])

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not args.debug:
  orig_stdout = sys.stdout
  f = open(os.path.join(log_dir, 'log.txt'),'w')
  sys.stdout= f

loss_func_mse = nn.MSELoss()
loss_raw = AverageMeter()
loss_lh = AverageMeter()

# Training
idx = 0
model.train()
meta_init_grads = []
meta_alpha_grads = []
for epoch in range(start_epoch, args.meta_epoch):
    loss = np.zeros(args.batch_size)
    lh_loss = np.zeros(args.batch_size)
    
    pbar = tqdm(total=len(train_batch))
    for j,(imgs) in enumerate(train_batch):
        meta_init_grads.clear()
        meta_alpha_grads.clear()
        imgs = Variable(imgs).cuda()

        for k in range(args.batch_size):
          
          init_g, alpha_g, loss[k], lh_loss[k], idx = train_init(model, meta_init, meta_alpha, loss_func_mse,
            imgs[k][:args.task_size//2,:12], imgs[k][args.task_size//2:,:12], imgs[k][:args.task_size//2,12:], imgs[k][args.task_size//2:,12:], idx, args)
          
          meta_init_grads.append(init_g)
          meta_alpha_grads.append(alpha_g)
        

        loss_raw.update(loss.mean().item(), 1)
        loss_lh.update(lh_loss.mean().item(), 1)

        
        meta_update(model, meta_init, meta_init_grads, meta_alpha, meta_alpha_grads,
                    meta_init_optimizer, meta_alpha_optimizer)

        pbar.set_postfix({
                      'Mpoch': '{0} {1}'.format(epoch+1, args.exp_dir),
                      'Lr': '{:.6f},{:.6f}'.format(meta_init_optimizer.param_groups[-1]['lr'], meta_alpha_optimizer.param_groups[-1]['lr']),
                      'loss': '{:.6f}({:.4f})'.format(loss.mean().item(), loss_raw.avg),
                      'lh_loss': '{:.6f}({:.4f})'.format(lh_loss.mean().item(), loss_lh.avg),
                    })
        pbar.update(1)

    print('----------------------------------------')
    print('Mpoch:', epoch+1)
    print('Lr: {:.6f}'.format(meta_init_optimizer.param_groups[-1]['lr'], meta_alpha_optimizer.param_groups[-1]['lr']))
    print('loss: {:.6f}({:.4f})'.format(loss.mean().item(), loss_raw.avg))
    print('lh_loss: {:.6f}({:.4f})'.format(lh_loss.mean().item(), loss_lh.avg))
    print('----------------------------------------')
    

    pbar.close()

    loss_raw.reset()
    loss_lh.reset()


    # Save the model and the memory items
    if epoch%100==0:
        state = {
              'epoch': epoch,
              'meta_init': meta_init,
              'meta_alpha': meta_alpha,
              'meta_init_optimizer' : meta_init_optimizer.state_dict(),
              'meta_alpha_optimizer': meta_alpha_optimizer.state_dict(),

            }
        torch.save(state, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))

print('Training is finished')
if not args.debug:
  sys.stdout = orig_stdout
  f.close()



