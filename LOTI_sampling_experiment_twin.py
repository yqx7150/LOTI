from operator_fza import forward, backward
import seaborn as sns
import matplotlib
import importlib
import os
import functools
from pixel import pixel3,pixel3_3
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import  LOTI_reconstruction_experiment_twin
from MyAdjointOperatorPropagation import my_adjoint_operator_propagation
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
import cv2
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization

from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from LOTI_reconstruction_experiment_twin import (ReverseDiffusionPredictor,
                                                 LangevinCorrector,
                                                 EulerMaruyamaPredictor,
                                                 AncestralSamplingPredictor,
                                                 NoneCorrector,
                                                 NonePredictor,
                                                 AnnealedLangevinDynamics)
import datasets
import time

time_begin = time.time()
# @title Load the score-based model
sde = 'VESDE'
sde_T= 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import church_ncsnpp_continuous as configs
  from configs.ve import church_ncsnpp_continuous_T as configs_T
  ckpt_filename = "/home/qgl/桌面/twin_image/MLDM-main（１）/MLDM_I/exp_train_pingole_image/checkpoints/checkpoint_43.pth" #(9:(20.2,0.5)
  ckpt_filename_T = "/home/qgl/桌面/twin_image/MLDM-main（１）/MLDM_I/exp_train_twin5/checkpoints/checkpoint_50.pth"  # (9:(20.2,0.5)
  config = configs.get_config()
  config_T = configs_T.get_config()
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sde_T = VESDE(sigma_min=config_T.model.sigma_min, sigma_max=config_T.model.sigma_max, N=config_T.model.num_scales)
  sampling_eps = 1e-5

batch_size =  1 #64#@param {"type":"integer"}

config.training.batch_size = batch_size
config_T.training.batch_size = batch_size

config.eval.batch_size = batch_size
config_T.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
sigmas_T= mutils.get_sigmas(config_T)

scaler = datasets.get_data_scaler(config)
scaler_T = datasets.get_data_scaler(config_T)

inverse_scaler = datasets.get_data_inverse_scaler(config)
inverse_scaler_T= datasets.get_data_inverse_scaler(config_T)

score_model = mutils.create_model(config)
score_model_T = mutils.create_model(config_T)

optimizer = get_optimizer(config, score_model.parameters())
optimizer_T = get_optimizer(config, score_model_T.parameters())

ema = ExponentialMovingAverage(score_model.parameters(),decay=config.model.ema_rate)
ema_T = ExponentialMovingAverage(score_model_T.parameters(),decay=config_T.model.ema_rate)

state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
state_T = dict(step=0, optimizer=optimizer_T,
             model=score_model_T, ema=ema_T)

state = restore_checkpoint(ckpt_filename, state, config.device)
state_T = restore_checkpoint(ckpt_filename_T, state_T, config_T.device)

ema.copy_to(score_model.parameters())
ema_T.copy_to(score_model_T.parameters())

#@title PC inpainting

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}



for j in range(0,1,1):
  img_ob_b = cv2.imread('/home/qgl/桌面/twin_image/MLDM-main（１）/MLDM_I/input/光照/手机聚光/Capture_00001.png', -1)
  img_ob_g = cv2.imread('/home/qgl/桌面/twin_image/MLDM-main（１）/MLDM_I/input/光照/手机聚光/Capture_00001.png', -1)
  img_ob_r = cv2.imread('/home/qgl/桌面/twin_image/MLDM-main（１）/MLDM_I/input/光照/手机聚光/Capture_00001.png', -1)
  print(img_ob_b.shape)
  Im = np.stack((img_ob_r[:, :], img_ob_g[:, :], img_ob_b[:, :]), axis=2)

  Xc1 = 2546 + 3
  Yc1 = 2004
  Xc2 = 3771
  Yc2 = 2017 - 6

  dp = 0.0038 * 3
  di = 1
  z1 = 400
  r1 = 0.3
  M = di / z1
  ri = (1 + M) * r1

  Nx, Ny = 256, 256

  fu_max, fv_max = 0.5 / dp, 0.5 / dp
  du, dv = 2 * fu_max / Nx, 2 * fv_max / Ny
  u, v = np.mgrid[-fu_max:fu_max:du, -fv_max:fv_max:dv]
  u = u.T
  v = v.T
  H1 = 1j * (np.exp(-1j * (np.dot(np.pi, ri ** 2)) * (u ** 2 + v ** 2)))
  H2 = 1j * (np.exp(-1j * (np.dot(np.pi, ri ** 2)) * (u ** 2 + v ** 2) + 1j * 0.5 * np.pi))

  H1 = np.array(H1, dtype=np.complex128)
  H2 = np.array(H2, dtype=np.complex128)

  a = 768
  I1_R = Im[round(Yc1 - a / 2) - 1:round(Yc1 + a / 2) - 1,
         round(Xc1 - a / 2) - 1:round(Xc1 + a / 2) - 1, 0]
  I1_G = Im[round(Yc1 - a / 2) - 1:round(Yc1 + a / 2) - 1,
         round(Xc1 - a / 2) - 1:round(Xc1 + a / 2) - 1, 1]
  I1_B = Im[round(Yc1 - a / 2) - 1:round(Yc1 + a / 2) - 1,
         round(Xc1 - a / 2) - 1:round(Xc1 + a / 2) - 1, 2]

  I2_R = Im[round(Yc2 - a / 2) - 1:round(Yc2 + a / 2) - 1,
         round(Xc2 - a / 2) - 1:round(Xc2 + a / 2) - 1, 0]
  I2_G = Im[round(Yc2 - a / 2) - 1:round(Yc2 + a / 2) - 1,
         round(Xc2 - a / 2) - 1:round(Xc2 + a / 2) - 1, 1]
  I2_B = Im[round(Yc2 - a / 2) - 1:round(Yc2 + a / 2) - 1,
         round(Xc2 - a / 2) - 1:round(Xc2 + a / 2) - 1, 2]

  I1 = np.stack((I1_R, I1_G, I1_B), axis=2)
  I2 = np.stack((I2_R, I2_G, I2_B), axis=2)
  I1 = pixel3(I1)
  I2 = pixel3(I2)

  # 对每个通道进行归一化
  def normalize_channel(channel):
      min_val = channel.min()
      max_val = channel.max()
      range_val = max_val - min_val

      return (channel - min_val) / range_val

  I1 = np.stack([normalize_channel(I1[:, :, i]) for i in range(3)], axis=2)
  I2 = np.stack([normalize_channel(I2[:, :, i]) for i in range(3)], axis=2)
  I1 = torch.from_numpy(I1)
  I2 = torch.from_numpy(I2)

  psnr_max_1=0
  for i in range(1):
    print('##################'+str(i)+'#######################')
     
    img_size = config.data.image_size                  #256
    channels = config.data.num_channels               #3
    shape = (batch_size, channels, img_size, img_size)   #(1,3,256,256)

    sampling_fn = LOTI_reconstruction_experiment_twin.get_pc_sampler(sde,sde_T,shape, predictor, corrector,
                                    inverse_scaler, snr, n_steps=n_steps,
                                    probability_flow=probability_flow,
                                    continuous=config.training.continuous,
                                    continuous_T=config_T.training.continuous,
                                    eps=sampling_eps, device=config.device)

    x,psnr_max,ssim_max = sampling_fn(score_model,score_model_T,H1,H2,I1,I2)                         #img:tensor(1,3,256,256)    H:numpy(256,256)  img_ob:tensor(256,256,3)


    cv2.imwrite('./Reconstruction_img.png',x*255)
    time_end = time.time()
    print('time:',time_end-time_begin)

      




