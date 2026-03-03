
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import LOTI_reconstruction_simulate_twin


import torch.nn as nn
import numpy as np

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
from LOTI_reconstruction_simulate_twin import (ReverseDiffusionPredictor,
                                               LangevinCorrector,
                                               EulerMaruyamaPredictor,
                                               AncestralSamplingPredictor,
                                               NoneCorrector,
                                               NonePredictor,
                                               AnnealedLangevinDynamics)
import datasets
import scipy.io as io
from operator_fza import forward,backward,forward_torch,backward_torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
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

#random_seed = 0 #@param {"type": "integer"}

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

psnr_result=[ ]
ssim_result=[ ]


input_folder = './input/sim' 
output_folder = './output/sim'  
os.makedirs(output_folder, exist_ok=True)


mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
mat_files.sort() 


psnr_results = []
psnr_1_results = []
ssim_results = []
ssim_1_results = []
time_results = []
k=0

for mat_file in mat_files:
    k+=1
    print(f"\n{'=' * 50}")
    print(f"Processing file: {mat_file}")
    print(f"{'=' * 50}")

    mat_path = os.path.join(input_folder, mat_file)
    mat_data = io.loadmat(mat_path)

    img = mat_data['Im']  
    img_ob1 = mat_data['I1']  
    img_ob2 = mat_data['I3']  


    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda().float()
    img_ob1 = torch.from_numpy(img_ob1).cuda().float()
    img_ob2 = torch.from_numpy(img_ob2).cuda().float()


    dp = 0.014
    di = 3
    z1 = 300
    r1 = 0.23#0.23
    M = di / z1
    ri = (1 + M) * r1

    NX, NY = 256, 256
    fu_max, fv_max = 0.5 / dp, 0.5 / dp
    du, dv = 2 * fu_max / NX, 2 * fv_max / NY
    u, v = np.mgrid[-fu_max:fu_max:du, -fv_max:fv_max:dv]
    u = u.T
    v = v.T

    H1 = 1j * np.exp(-1j * (np.pi * ri ** 2) * (u ** 2 + v ** 2))
    H2 = 1j * np.exp(-1j * (np.pi * ri ** 2) * (u ** 2 + v ** 2) + 1j * 0.5 * np.pi)
    H1 = np.array(H1, dtype=np.complex128)
    H2 = np.array(H2, dtype=np.complex128)


    img_size = config.data.image_size
    channels = config.data.num_channels
    shape = (batch_size, channels, img_size, img_size)
    start_time = time.time()
    sampling_fn = LOTI_reconstruction_simulate_twin.get_pc_sampler(sde,sde_T,shape, predictor, corrector,
                                    inverse_scaler, snr, n_steps=n_steps,
                                    probability_flow=probability_flow,
                                    continuous=config.training.continuous,
                                    continuous_T=config_T.training.continuous,
                                    eps=sampling_eps, device=config.device)

    x, psnr_max, ssim_max, psnr_1_max, ssim_1_max = sampling_fn(score_model,score_model_T,img,H1,H2,img_ob1,img_ob2,mat_file)
    elapsed_time = time.time() - start_time

    filename = os.path.splitext(mat_file)[0]
 
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / (x_max - x_min)
    output_path = os.path.join(output_folder, f"{filename}_recon.png")
    cv2.imwrite(output_path, x_normalized * 255)

    psnr_results.append(psnr_max)
    psnr_1_results.append(psnr_1_max)
    ssim_results.append(ssim_max)
    ssim_1_results.append(ssim_1_max)
    time_results.append(elapsed_time)

    print(f"File {mat_file} processed in {elapsed_time:.2f} seconds")
    print(f"PSNR: {psnr_max:.4f}, SSIM: {ssim_max:.4f}")

avg_psnr = np.mean(psnr_results) if psnr_results else 0
avg_psnr_1 = np.mean(psnr_1_results) if psnr_1_results else 0
avg_ssim = np.mean(ssim_results) if ssim_results else 0
avg_ssim_1 = np.mean(ssim_1_results) if ssim_1_results else 0
avg_time = np.mean(time_results) if time_results else 0

print(f"Average Results for {len(mat_files)} files:")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average PSNR_1: {avg_psnr_1:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average SSIM_1: {avg_ssim_1:.4f}")
print(f"Average Processing Time: {avg_time:.2f} seconds")


results_df = pd.DataFrame({
    'File': mat_files,
    'PSNR': psnr_results,
    'PSNR_1': psnr_1_results,
    'SSIM': ssim_results,
    'SSIM_1': ssim_1_results,
    'Time(s)': time_results
})


avg_row = pd.DataFrame({
    'File': ['Average'],
    'PSNR': [avg_psnr],
    'PSNR_1': [avg_psnr_1],
    'SSIM': [avg_ssim],
    'SSIM_1': [avg_ssim_1],
    'Time(s)': [avg_time]
})
results_df = pd.concat([results_df, avg_row], ignore_index=True)


results_csv = os.path.join(output_folder, 'reconstruction_results.csv')
results_df.to_csv(results_csv, index=False)


with open(os.path.join(output_folder, 'average_results.txt'), 'w') as f:
    f.write(f"Average Results for {len(mat_files)} files:\n")
    f.write(f"Average PSNR: {avg_psnr:.4f}\n")
    f.write(f"Average PSNR_1: {avg_psnr_1:.4f}\n")
    f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    f.write(f"Average SSIM_1: {avg_ssim_1:.4f}\n")
    f.write(f"Average Processing Time: {avg_time:.2f} seconds\n")
    f.write(f"Total Processing Time: {sum(time_results):.2f} seconds\n")

print("\nProcessing completed!")
print(f"Results saved to: {output_folder}")
print(f"Total time: {time.time() - time_begin:.2f} seconds")
      




