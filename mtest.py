# -*- coding: UTF-8 -*-
'''
@Author: Chuangbin Chen
@Date: 2020-01-02 22:37:14
@LastEditTime : 2020-01-02 23:28:31
@LastEditors  : Do not edit
@Description: 
'''

#%%
import torch
from loader.loader_utils import png_reader_32bit, png_reader_uint8
import os 
from os.path import join as pjoin
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt

def read_normal(img_name,img_size=(256,320)):
    lb_path_nx = img_name
    lb_path_ny = img_name.replace('_mesh_nx', '_mesh_ny')
    lb_path_nz = img_name.replace('_mesh_nx', '_mesh_nz')

    lbx = png_reader_32bit(lb_path_nx, img_size)
    lby = png_reader_32bit(lb_path_ny, img_size)
    lbz = png_reader_32bit(lb_path_nz, img_size)

    lbx = lbx.astype(float)
    lby = lby.astype(float)
    lbz = lbz.astype(float)  
    lbx = lbx/65535
    lby = lby/65535
    lbz = lbz/65535
    
    # Resize scales masks from 0 ~ 1
    # mask是normal的平方，浮点数，并非只有[0,1]二元mask
    mask = np.power(lbx,2) + np.power(lby,2) + np.power(lbz,2)
    mask = (mask>0.001).astype(float)
    
    #file holes            
    lbx[mask == 0] = 0.5
    lby[mask == 0] = 0.5
    lbz[mask == 0] = 0.5
    
    lb = np.concatenate((lbx[:,:,np.newaxis], lby[:,:,np.newaxis],lbz[:,:,np.newaxis]), axis = 2)
    # 缩放到[-1,1]
    lb = 2*lb-1 
    return lb
#%%
if __name__ == '__main__':
    # img_size=(256,320)
    normal = read_normal('/home/lab/data1/chuangbin/dataSet/matterport/v1/scans/17DRP5sb8fy/mesh_images/f4d03f729dfc49068db327584455e975_d1_2_mesh_nx.png')
    plt.figure()
    plt.imshow(normal)
    plt.show()
    io.savemat('./result/normal.mat', {'normal': normal})


# %%
