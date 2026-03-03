# LOTI

**Paper**: High-quality FZA lensless imaging via joint generation of original-twin images

**Authors**: QI YU†, MINGCHUN HUANG†, YUHAO WANG, WENBO WAN AND QIEGEN LIU

Date : Jan-30-2026  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
School of Information Engineering, Nanchang University. 

Lensless imaging possesses the advantages of functional flexibility and high portability. The inherent twin-image problem in Fresnel zone aperture (FZA) lensless imaging cannot be eliminated using conventional image reconstruction methods based on spatial sparsity priors. A low-cost FZA encoder with orthogonal phases is used to construct a lensless imaging system, and high-quality FZA lensless imaging via joint generation of original-twin images (LOTI) is proposed. Image generation is performed in a high-dimensional solution space of original-twin images, separating the twin image from the reconstructed content and producing an artifact-free original image. LOTI demonstrates strong robustness to different mask fabrication methods, phase encoding schemes, target distances, and distance mismatch. Simulative and experimental validation demonstrate LOTI effectively suppresses twin-image artifacts in the reconstructed images while preserving textural details and color fidelity. Quantitative evaluations further confirm the superior performance, with an average peak signal-to-noise ratio (PSNR) of 24.37 dB and structural similarity index measure (SSIM) of 0.92 within the regions of interest.


## The schematic diagram of FZA lensless imaging principle.
<div align="center"><img src="https://github.com/yqx7150/MLDM/blob/main/Figs/fig1.png"> </div>

## Reconstruction flow chart of MLDM-I algorithm.
<div align="center"><img src="https://github.com/yqx7150/MLDM/blob/main/Figs/fig3.png"> </div>

## Reconstruction flow chart of MLDM-II algorithm.
<div align="center"><img src="https://github.com/yqx7150/MLDM/blob/main/Figs/fig5.png"> </div>

## The multi-phase lensless imaging system.
<div align="center"><img src="https://github.com/yqx7150/MLDM/blob/main/Figs/fig8.png"> </div>

## Visual comparison of reconstructed images on optical experiment. (a) Ground Truth (b) BP (c) CS (d) LSGM (e) ADMM (f) MLDM-I (g) MLDM-II.
<div align="center"><img src="https://github.com/yqx7150/MLDM/blob/main/Figs/fig9.png"> </div>


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
MLDM_I : We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud] (https://pan.baidu.com/s/1CX7xCh1uJl-h5ZojO5SkNQ?pwd=hdtp) Extract the code (hdtp)

MLDM_II : We provide pretrained checkpoints. You can download pretrained models from  [Baidu cloud] (https://pan.baidu.com/s/1-0WZnOfjQ5eiTjk1aO93PQ?pwd=taf4) Extract the code (taf4)

## Dataset

The dataset used to train the model in this experiment is  LSUN-bedroom and  LSUN-church.

place the dataset in the train file under the church folder.

## Train:

python main.py --config=configs/ve/church_ncsnpp_continuous.py  --workdir=exp_train_church_max1_N1000 --mode=train --eval_folder=result


## Test:

Simulate : python MLDM_reconstruction_simulate.py

Experiment : python MLDM_reconstruction_experiment.py


## Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.


## Other Related Projects
  * Lens-less imaging via score-based generative model  
[<font size=5>**[Paper]**</font>](https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/LSGM)

  * Imaging through scattering media via generative diffusion model  
[<font size=5>**[Paper]**</font>](https://pubs.aip.org/aip/apl/article/124/5/051101/3176612/Imaging-through-scattering-media-via-generative)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/ISDM)

  * Dual-domain Mean-reverting Diffusion Model-enhanced Temporal Compressive Coherent Diffraction Imaging  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.517567)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DMDTC)   
    
  * High-resolution iterative reconstruction at extremely low sampling rate for Fourier single-pixel imaging via diffusion model  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.510692)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/FSPI-DM)

  * Real-time intelligent 3D holographic photography for real-world scenarios  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.529107)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Intelligent-3D-holography)
