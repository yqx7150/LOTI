# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

from configs.default_lsun_configs_T import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.category = 'church_outdoor'

  # model
  model = config.model
  model.name = 'ncsnpp'
  #model.sigma_max = 380
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config



"""
这段代码定义了一个名为 `get_config` 的函数，它获取默认配置并进行了一些修改。

以下是修改部分的详细说明：

- 在 `training` 部分：
    - 指定了使用的随机微分方程（SDE）为 `vesde` 。
    - 保持训练为连续模式。

- 在 `sampling` 部分：
    - 设定采样方法为 `pc`（预测器-校正器）。
    - 指定预测器为 `reverse_diffusion` 。
    - 指定校正器为 `langevin` 。

- 在 `data` 部分：
    - 指定数据类别为 `church_outdoor` 。

- 在 `model` 部分：
    - 设定模型名称为 `ncsnpp` 。
    - 开启了一些与模型结构和参数相关的设置，如 `scale_by_sigma` 、 `ema_rate` 、各种与模型层和操作相关的参数等。

总的来说，这个函数返回了一个经过定制修改的配置对象，用于特定的应用场景。
"""