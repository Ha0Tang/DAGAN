[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/DAGAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.0.0-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Ha0Tang/DAGAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Ha0Tang/DAGAN/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

## Contents
  - [DAGAN](#Semantic-Image-Synthesis-with-DAGAN)
  - [Installation](#Installation)
  - [Dataset Preparation](#Dataset-Preparation)
  - [Generating Images Using Pretrained Model](#Generating-Images-Using-Pretrained-Model)
  - [Train and Test New Models](#Train-and-Test-New-Models)
  - [Evaluation](#Evaluation)
  - [Acknowledgments](#Acknowledgments)
  - [Related Projects](#Related-Projects)
  - [Citation](#Citation)
  - [Contributions](#Contributions)

## Semantic Image Synthesis with DAGAN

**Dual Attention GANs for Semantic Image Synthesis (Coming Soon!)**  
[Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>1</sup>, [Song Bai](http://songbai.site/)<sup>2</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>13</sup>. <br> 
<sup>1</sup>University of Trento, Italy, <sup>2</sup>University of Oxford, UK, <sup>3</sup>Huawei Research Ireland, Ireland. <br>
In [ACM MM 2020](https://2020.acmmm.org/). <br>
The repository offers the official implementation of our paper in PyTorch.

Also see our CVPR 2020 paper [Local Class-Specific and Global Image-Level Generative Adversarial Networks for Semantic-Guided Scene Generation](https://github.com/Ha0Tang/LGGAN).

### Framework
<img src='./imgs/method.jpg' width=1200>

### Results of Generated Images

#### Cityscapes (512×256)
<img src='./imgs/city_results.jpg' width=1200>

#### Facades (1024×1024)
<img src='./imgs/facades_results.jpg' width=1200>

#### ADE20K (256×256)
<img src='./imgs/ade_results.jpg' width=1200>

#### CelebAMask-HQ (512×512)
<img src='./imgs/celeba_results.jpg' width=1200>

### Results of Generated Segmenation Maps

<img src='./imgs/seg.jpg' width=1200>

### [License](./LICENSE.md)

Copyright (C) 2019 University of Trento, Italy.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [hao.tang@unitn.it](hao.tang@unitn.it).

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/DAGAN
cd DAGAN/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

To reproduce the results reported in the paper, you would need an NVIDIA DGX1 machine with 8 V100 GPUs.

## Dataset Preparation

For Facades, CelebAMask-HQ, Cityscapes or ADE20K, the datasets must be downloaded beforehand. Please download them on the respective webpages. In the case of COCO-stuff, we put a few sample images in this code repo.
- Facades: [here](http://cmp.felk.cvut.cz/~tylecr1/facade/).
- CelebAMask-HQ: [here](https://github.com/switchablenorms/CelebAMask-HQ).
- Cityscapes: [here](https://www.cityscapes-dataset.com/).
- ADE20K: [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
For Cityscapes or ADE20K, please refer to [GauGAN](https://github.com/NVlabs/SPADE) for more details.

## Generating Images Using Pretrained Model

## Train and Test New Models

## Evaluation

## Related Projects
**[EdgeGAN](https://github.com/Ha0Tang/EdgeGAN) | [LGGAN](https://github.com/Ha0Tang/LGGAN) | [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) | [PanoGAN](https://github.com/sswuai/PanoGAN) | [Guided-I2I-Translation-Papers](https://github.com/Ha0Tang/Guided-I2I-Translation-Papers)**

## Citation
If you use this code for your research, please cite our papers.

EdgeGAN
```
@article{tang2020edge,
  title={Edge Guided GANs with Semantic Preserving for Semantic Image Synthesis},
  author={Tang, Hao and Qi, Xiaojuan and Xu, Dan and Torr, Philip HS and Sebe, Nicu},
  journal={arXiv preprint arXiv:2003.13898},
  year={2020}
}
```

LGGAN
```
@inproceedings{tang2019local,
  title={Local Class-Specific and Global Image-Level Generative Adversarial Networks for Semantic-Guided Scene Generation},
  author={Tang, Hao and Xu, Dan and Yan, Yan and Torr, Philip HS and Sebe, Nicu},
  booktitle={CVPR},
  year={2020}
}
```

SelectionGAN
```
@inproceedings{tang2019multi,
  title={Multi-channel attention selection gan with cascaded semantic guidance for cross-view image translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Wang, Yanzhi and Corso, Jason J and Yan, Yan},
  booktitle={CVPR},
  year={2019}
}

@article{tang2020multi,
  title={Multi-channel attention selection gans for guided image-to-image translation},
  author={Tang, Hao and Xu, Dan and Yan, Yan and Corso, Jason J and Torr, Philip HS and Sebe, Nicu},
  journal={arXiv preprint arXiv:2002.01048},
  year={2020}
}
```

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([hao.tang@unitn.it](hao.tang@unitn.it)).