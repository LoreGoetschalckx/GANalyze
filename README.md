![Visualizations produced by the proposed GANalyze framework}. The middle columns represent generated images serving as the original seed. The originals are then modified to be characterized more (right) or less (left) by a given property of interest (memorability, aesthetics, or emotional valence). The images' respective property scores are presented in their top left corner.](http://ganalyze.csail.mit.edu/img/teaser2.jpg)


# GANalyze

The authors' official implementation of *GANalyze*, a framework for studying cognitive properties such as memorability, aesthetics, and emotional valence using genenerative models.

## Overview
- [Requirements and Installation](#requirements-and-installation)
- [Quick Start](#quick-start)
- [Reference](#reference)

## Requirements and Installation

We provide both Tensorflow and Pytorch implementations. You should feel free to use either or both depending on your prefences. Although this code may be compatible with the latest versions of these libraries, we have officially tested it with the following:

- [PyTorch](https://pytorch.org/get-started/locally/) >= 0.4 (1.1.0) and [torchvision](https://github.com/pytorch/vision) >= 0.2.2 (0.3.0)
- [Tensorflow](https://www.tensorflow.org/install) 1.12.0 and [tensorflow_hub](https://www.tensorflow.org/hub) 0.1.1 (for pretrained BigGANs)
- numpy, scipy, PIL

We recommend referring to the native documentation for more detailed installation instructions. However, if you are installing on a linux server running Ubuntu, the following commands will likely suffice:

```bash
# Tensorflow 1.12 with GPU support (highly recommended)
pip install tensorflow-gpu==1.12
# Tensorflow hub (for pretrained BigGAN modules)
pip install tensorflow_hub==0.1.1

# PyTorch and torchvision with latest version of cuda toolkit.
# Note: Anaconda is the recommended package manager for PyTorch.
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Finally, to clone this repo, run:

```bash
git clone https://github.com/LoreGoetschalckx/GANalyze.git
cd GANalyze
```

## Quick Start

We provide both training and testing scripts, which can be used to reproduce the results shown in the paper. These scripts accept a pretrained GAN Generator (e.g. BigGAN) and a pretrained Assessor (e.g. MemNet) to train/test a Transformer module, the third component of the the GANalyze framework. Below, we demonstrate how the Tensorflow scripts can be used to reproduce the memorability ganalysis from the paper. Note, the PyTorch scripts behave in a nearly identical manner.

**Downloading Pretrained Models:**
In order to effectively use this code, you can first download pretrained generators and assessors hosted on the GANalyze project page. We provide utility scripts for downloading both Tensorflow and/or PyTorch models, which can be called with the following:

Tensorflow:
```bash
cd tensorflow; sh download_pretrained.sh
```

PyTorch:
```bash
cd pytorch; sh download_pretrained.sh
```
These will populate the appropriate directores with models weights that will be used with the scripts demonstrated below.

**Training:**

To begin training, we first need to specifiy the GAN generator architecture/model and the assessor function. Below we use a 256px BigGAN trained on ImageNet and the standard pretrained MemNet. Next, specify the transformer class `OneDirection` and its keyword arguments (`None` in this case). Finally, we specify some training parameters such as the upper and lower bounds on `alpha`, training duration, if/when to resume, etc. (see options at the top of the script for a complete list). Training can be initiated with the following:

```bash
python train_tf.py \
 --generator_arch biggan --generator_model biggan256 \
 --assessor memnet \
 --transformer OneDirection None \
 --train_alpha_a -0.5 --train_alpha_b 0.5 \
 --gpu_id 0 --num_samples 400000 --checkpoint_resume 0
```

During training, this script will store checkpoints of the transformer module in a subdirectory of `checkpoints` whose full path depends on the experiment's configuration. For the command above, checkpoints are stored in `checkpoints/biggan__biggan256/memnet/OneDirection/<commit_hash>`, where `<commit_hash>` is the active commit's hash assigned by git (if you checkout a new/different commit of the code, it will be stored in a new directory as to not overwrite previous experiments).

**Testing:**

Once you have trained a transformer module using the training script, *GANalyze* what your model has learned by generating sample sequences where alpha has been systematically varied given a starting seed image.

```bash
python test_tf.py \
--alpha 0.1 --test_truncation 1 \
--checkpoint_dir checkpoints/biggan__biggan256/memnet/OneDirection_None/<commit_hash> \
--checkpoint 400000 \
--gpu_id 0
```

This script will populate a subdirectory of `./output` (uses same naming convention as training) with samples have been transformed by several steps of size `alpha`.

## Reference

If you use this code or work, please cite:

L. Goetschalckx, A. Andonian, A. Oliva, and P. Isola. GANalyze: Toward Visiual Definitions of Cognitive Image Properties. arXiv:1906.10112, 2019.

The following is a BibTeX reference entry:

```markdown
@article{
  title={GANalyze: Toward Visual Definitions of Cognitive Image Properties},
  author={Goetschalckx, Lore and Andonian, Alex and Oliva, Aude and Isola, Phillip},
  journal={arXiv preprint arXiv:1906.10112},
  year={2019}
}
```
