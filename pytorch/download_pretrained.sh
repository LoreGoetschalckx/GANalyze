#!/usr/bin/env bash

echo "Downloading EmoNet weights"
wget -P assessors http://ganalyze.csail.mit.edu/models/EmoNet_valence_moments_resnet50_5_best.pth.tar

echo "Downloading BigGAN weights"
wget -P generators http://ganalyze.csail.mit.edu/models/biggan-128.pth
wget -P generators http://ganalyze.csail.mit.edu/models/biggan-256.pth
wget -P generators http://ganalyze.csail.mit.edu/models/biggan-512.pth

