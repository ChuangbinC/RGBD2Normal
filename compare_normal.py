'''
@Author: Chuangbin Chen
@Date: 2019-11-13 22:20:51
@LastEditTime: 2019-11-13 22:23:57
@LastEditors: Do not edit
@Description: 
'''
import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from os.path import join as pjoin
import scipy.io as io

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model, get_lossfun
from loader import get_data_path, get_loader
from pre_trained import get_premodel
from utils import norm_imsave, change_channel
from models.eval import eval_normal_pixel, eval_print
from loader.loader_utils import png_reader_32bit, png_reader_uint8





if __name__ == "__main__":
    compareNormal()