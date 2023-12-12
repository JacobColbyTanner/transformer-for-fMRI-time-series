
from scipy.io import loadmat, savemat
import numpy as np
import model as net
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

import h5py

def print_structure(name, obj):
    print(name, type(obj))


# Open the file
data = loadmat('/N/project/networkRNNs/HCP_movie_stimulus/HCP7t_1000_all_ts.mat')

print(data["all_ts"].shape)

print(data["all_ts"][0,0])