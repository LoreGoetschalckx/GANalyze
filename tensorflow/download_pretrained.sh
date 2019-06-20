#!/usr/bin/env bash

echo "Downloading AestheticsNet weights"
wget -P assessors http://ganalyze.csail.mit.edu/models/aestheticsnet_state_dict.p

echo "Downloading MemNet weights"
wget -P assessors http://ganalyze.csail.mit.edu/models/memnet_state_dict.p
wget -P assessors http://ganalyze.csail.mit.edu/models/memnet_mean.mat

wget -P assessors http://ganalyze.csail.mit.edu/models/mean_AADB_regression_warp256.binaryproto
wget -P assessors http://ganalyze.csail.mit.edu/models/mean_AADB_regression_warp256_lore.npy
