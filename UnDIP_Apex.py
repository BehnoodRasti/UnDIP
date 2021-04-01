# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:12:43 2021

@author: behnood
"""

from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib inline

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from utils.denoising_utils import *

from skimage._shared import *
from skimage.util import *
from skimage.metrics.simple_metrics import _as_floats
from skimage.metrics.simple_metrics import mean_squared_error

from UtilityMine import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = False
import scipy.io
#%%
fname2  = "C:/Users/Behnood/Data/crop/5class/Y_clean.mat"
mat2 = scipy.io.loadmat(fname2)
img_np_gt = mat2["Y_clean"]
img_np_gt = img_np_gt.transpose(2,0,1)
[p1, nr1, nc1] = img_np_gt.shape
#%%
import pymf
from pymf.sivm import *
from pymf.chnmf import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
tol2=5
TimeSec=np.zeros((tol2,1))
for fj in tqdm(range(tol2)):
    start_time = time.time()
        #%%
    img_noisy_np=img_np_gt
    img_resh=np.reshape(img_np_gt,(p1,nr1*nc1))
    V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
    PC=np.diag(SS)@U
    rmax=4
    img_resh_DN=V[:,:rmax-1]@PC[:rmax-1,:]
    img_resh_np_clip=np.clip(img_resh_DN, 0, 1)
    II,III = Endmember_extract(img_resh_np_clip,rmax-1)
    E_np1=img_resh_np_clip[:,II]
    E_np1=np.concatenate((E_np1, np.zeros((285,1))), axis=1)
    #%% Set up Simulated 
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    need_bias=True
    OPT_OVER = 'net' # 'net,input'
    
    # 
    LR1 = 0.001
    
    OPTIMIZER1='adam' # 'RMSprop'#'adam' # 'LBFGS'
    show_every = 100
    exp_weight=0.99
    
    num_iter1 = 3000
    input_depth = rmax#img_noisy_np.shape[0]
    class CAE_AbEst(nn.Module):
        def __init__(self):
            super(CAE_AbEst, self).__init__()
            # encoding layers
            self.conv1 = nn.Sequential(
                skip(
                        input_depth, rmax,
                        # num_channels_down = [8, 16, 32, 64, 128], 
                        # num_channels_up   = [8, 16, 32, 64, 128],
                        # num_channels_skip = [4, 4, 4, 4, 4], 
                        num_channels_down = [ 256],
                        num_channels_up =   [ 256],
                        num_channels_skip =    [ 4],  
                        filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
                        upsample_mode='bilinear', # downsample_mode='avg',
                        need1x1_up=True,
                        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
            )
    
        def forward(self, x):
            x = self.conv1(x)
            return x
    
    net1 = CAE_AbEst()
    net1.cuda()
    
    # Compute number of parameters
    s  = sum([np.prod(list(p11.size())) for p11 in net1.parameters()]); 
    print ('Number of params: %d' % s)
    
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    # if fk==0:
    net_input1 = get_noise(input_depth, INPUT,
      (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()
    E_torch = np_to_torch(E_np1).type(dtype)
    #
    #%%
    net_input_saved = net_input1.detach().clone()
    noise = net_input1.detach().clone()
    out_avg = None
    out_HR_avg= None
    last_net = None
    RMSE_LR_last = 0
    
    i = 0
    def closure1():
        
        global i, out_avg,out_HR_np,out_HR_avg
        out_LR = net1(net_input1)
        out_HR=torch.mm(E_torch.view(p1,rmax),out_LR.view(rmax,nr1*nc1))
        # Smoothing
        if out_avg is None:
            out_avg = out_LR.detach()
            out_HR_avg = out_HR.detach()
        else:
            out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)
            out_HR_avg = out_HR_avg * exp_weight + out_HR.detach() * (1 - exp_weight)
          
        out_HR=out_HR.view((1,p1,nr1,nc1))
        total_loss = mse(img_noisy_torch, out_HR)
        total_loss.backward()
        print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        i += 1
    
        return total_loss
    
    p11 = get_params(OPT_OVER, net1, net_input1)
    optimize(OPTIMIZER1, p11, closure1, LR1, num_iter1)
