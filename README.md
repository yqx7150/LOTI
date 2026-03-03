# LOTI

**Paper**: High-quality FZA lensless imaging via joint generation of original-twin images

**Authors**: QI YU†, MINGCHUN HUANG†, YUHAO WANG, WENBO WAN AND QIEGEN LIU

Date : Jan-30-2026  
The code and the algorithm are for non-comercial use only.  
School of Information Engineering, Nanchang University. 

Lensless imaging possesses the advantages of functional flexibility and high portability. The inherent twin-image problem in Fresnel zone aperture (FZA) lensless imaging cannot be eliminated using conventional image reconstruction methods based on spatial sparsity priors. A low-cost FZA encoder with orthogonal phases is used to construct a lensless imaging system, and high-quality FZA lensless imaging via joint generation of original-twin images (LOTI) is proposed. Image generation is performed in a high-dimensional solution space of original-twin images, separating the twin image from the reconstructed content and producing an artifact-free original image. LOTI demonstrates strong robustness to different mask fabrication methods, phase encoding schemes, target distances, and distance mismatch. Simulative and experimental validation demonstrate LOTI effectively suppresses twin-image artifacts in the reconstructed images while preserving textural details and color fidelity. Quantitative evaluations further confirm the superior performance, with an average peak signal-to-noise ratio (PSNR) of 24.37 dB and structural similarity index measure (SSIM) of 0.92 within the regions of interest.

## The schematic diagram of FZA lensless imaging principle.
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.1.png"> </div>

## Joint learning of prior distributions in the high-dimensional solution space.
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.3.png"> </div>               

## Generation in high-dimensional joint solution space.
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.4.png"> </div>    

## Comprehensive comparison of results in simulative validation. (a) Ground Truth. (b) BP. (c) CS. (d) MLDM. (e) LOTI. 
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.5.png"> </div>    

## Comprehensive comparison of results in generalization validation. (a) Ground Truth. (b) BP. (c) CS. (d) MLDM. (e) LOTI. 
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.6.png"> </div>    

## Lensless imaging system encoded by orthogonal-phase FZA mask.
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.7.png"> </div>

## Comprehensive comparison of results in experimental validation. (a) Ground Truth. (b) BP. (c) CS. (d) MLDM. (e) LOTI. 
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.8.png"> </div>

## Reconstruction results under real-world scenarios. (a) Ground Truth. (b) BP. (c) CS. (d) MLDM. (e) LOTI. 
<div align="center"><img src="https://github.com/yqx7150/LOTI/blob/main/Figs/Fig.9.png"> </div>


## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud] (https://pan.baidu.com/s/1bNDjx4YN2bf8Lf8x98XKWg?pwd=8c44) Extract the code (8c44)

## Dataset

The dataset used to train the model in this experiment is  LSUN-bedroom and  LSUN-church.

place the dataset in the train file under the church folder.

## Train:

SNO:python main.py --config=configs/ve/church_ncsnpp_continuous.py  --workdir=exp_train_pingole_image --mode=train --eval_folder=result
SNT:python main.py --config=configs/ve/church_ncsnpp_continuous.py  --workdir=exp_train_twin_image --mode=train --eval_folder=result


## Test:

Simulate : python LOTI_reconstruction_simulate_twin.py

Experiment : python LOTI_reconstruction_experiment_twin.py

