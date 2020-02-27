# -*- coding: UTF-8 -*-
#%% 
import sys
sys.path.insert(0, '..')

import os
from os.path import join as pjoin
import collections
import json
import torch
import torch.nn.functional as F
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import scipy.misc as misc

from mpl_toolkits.axes_grid1 import make_axes_locatable

from loader.loader_utils import png_reader_32bit, png_reader_uint8
# from loader_utils import png_reader_32bit, png_reader_uint8
# from .loader_utils import png_reader_32bit, png_reader_uint8
from models.eval import eval_normal_pixel_me
from tqdm import tqdm
from torch.utils import data
import time 
from utils import change_channel_batch
# 加载matterport数据集

# 读取数据集路径
def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

class mtLoader(data.Dataset):
    """Data loader for the MatterPort3D dataset.

    """
    def __init__(self, root, split='train', img_size=(256,320), img_norm=True,mono=False,add_normal=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_norm = img_norm
        self.mono = mono
        self.add_normal = add_normal
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)

        
        # 设置正态分布的 均值和标准差
        # 均值在 0°到 35°之间  0.17 ～ 0.61
        # 标准差在 2°到12°之间  0.03 ～ 0.21
        np.random.seed(1)
        self.mu = np.random.uniform(0,0.61,105432)
        np.random.seed(2)
        self.sigma = np.random.uniform(0.03,0.21,105432)

        if(self.mono):
            print("Loading mono image dataset!")

        # for split in ['train', 'test', 'testsmall','small_100']:
        for split in ['train', 'test', 'testsmall','small_10']:
            path = pjoin('./datalist', 'mp_' + split + '_list.txt')
            # path = pjoin('../datalist', 'mp_' + split + '_list.txt')

            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def convert_rgb_mono(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = np.clip(np.stack([gray,gray,gray],axis=2),0,255).astype(np.uint8)
        return gray

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name_base = self.files[self.split][index]
        im_path = pjoin(self.root,  im_name_base)

        # 加载原始深度数据 undistorted_depth_images
        im_name = im_name_base.replace('_i', '_d')
        # 这里undistorted_color_dmages 是因为上面将_i换成 _d
        im_name = im_name.replace('undistorted_color_dmages', 'undistorted_depth_images')
        im_name = im_name.replace('.jpg', '.png')
        depth_path = pjoin(self.root, im_name)
        
        # starttime = time.time()
        # render_normal 改为 mesh_image
        im_name = im_name_base.replace('_i', '_d')
        # im_name = im_name.replace('undistorted_color_images', 'render_normal')    
        im_name = im_name.replace('undistorted_color_dmages', 'mesh_images')                     
        lb_path_nx = pjoin(self.root,  im_name.replace('.jpg', '_mesh_nx.png'))
        lb_path_ny = pjoin(self.root,  im_name.replace('.jpg', '_mesh_ny.png')) 
        lb_path_nz = pjoin(self.root,  im_name.replace('.jpg', '_mesh_nz.png')) 

        # 加载 _mesh_depth 数据集
        im_name = im_name_base.replace('_i', '_d')
        # im_name = im_name.replace('undistorted_color_images', 'render_depth')  
        im_name = im_name.replace('undistorted_color_dmages', 'mesh_images')     
        meshdepth_path = pjoin(self.root,  im_name.replace('.jpg', '_mesh_depth.png'))             

        im = png_reader_uint8(im_path, self.img_size)#uint8
        rawdepth = png_reader_32bit(depth_path, self.img_size)   #32bit uint
        lbx = png_reader_32bit(lb_path_nx, self.img_size)
        lby = png_reader_32bit(lb_path_ny, self.img_size)
        lbz = png_reader_32bit(lb_path_nz, self.img_size)
        meshdepth = png_reader_32bit(meshdepth_path, self.img_size)  

        if(self.mono):
            im = self.convert_rgb_mono(im)
        im = im.astype(float) 
        rawdepth = rawdepth.astype(float)     
        meshdepth = meshdepth.astype(float)      

        if self.img_norm:
            # Resize scales images from -0.5 ~ 0.5
            im = (im-128) / 255
            if(self.add_normal):
                # resize  [-1,1]
                # lbx = lbx/32768 - 1
                # lby = lby/32768 - 1
                # lbz = lbz/32768 - 1

                lbx = lbx/65535
                lby = lby/65535
                lbz = lbz/65535

                # normal 中会存在远处法线为空的现象
                mask = np.power(lbx,2) + np.power(lby,2) + np.power(lbz,2)
                zero_mask = (mask<0.001).astype(float)   
                # if(np.sum(zero_mask) > 8192):
                    # print('{} has {} bad points'.format(im_name,np.sum(zero_mask)))
                mask = (mask>0.001).astype(float)   
                lbx[mask == 0] = 0.5
                lby[mask == 0] = 0.5
                lbz[mask == 0] = 0.5

                lb = np.concatenate((lbx[:,:,np.newaxis], lby[:,:,np.newaxis], lbz[:,:,np.newaxis]), axis = 2)
                lb = 2*lb-1
            # original code
            else:
                # Resize scales labels from 0 ~ 1
                # 因为是16bit 所以有正负
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
                
                # lb = [lbx,1-lbz,lby] 1-lbz 表示相反
                lb = np.concatenate((lbx[:,:,np.newaxis], 1-lbz[:,:,np.newaxis], lby[:,:,np.newaxis]), axis = 2)
                # 缩放到[-1,1]
                lb = 2*lb-1


            # Resize scales valid, devide by mean value           
            rawdepth = rawdepth/40000
            meshdepth = meshdepth/40000
            # Get valid from rawdepth
            if self.add_normal:
                valid = np.ones(lb.shape).astype(float)
            else:
                valid = (rawdepth>0.0001).astype(float)

        if self.add_normal:

            len = np.linalg.norm(lb,ord=2,axis=2)
            # z/len
            b = np.divide(lb[:,:,2],len,out=np.zeros_like(len), where=len!=0.0)
            b = np.clip(b,-1,1)
            zenithAngle = np.arccos(b) 
            a = np.sin(zenithAngle)*len
            #y
            b = np.divide(lb[:,:,0],np.sin(zenithAngle)*len,out=np.zeros_like(a), where=a!=0.0)
            b = np.clip(b,-1,1)
            azimuthAngle = np.arccos(b)
            # 设置随机种子
            np.random.seed(0)
            noise = np.random.normal(self.mu[index],self.sigma[index],size=zenithAngle.shape)
            # noise[mask == 0] = 0
            # print((mask == 0).shape)
            mask_1 = (lb[:,:,1] < 0) * (lb[:,:,0] < 0) 
            azimuthAngle[mask_1] = 2*np.pi - azimuthAngle[mask_1]
            mask_1 = (lb[:,:,1] < 0) * (lb[:,:,0] > 0)
            azimuthAngle[mask_1] = 2*np.pi - azimuthAngle[mask_1]
            zenithAngle_original = zenithAngle

            zenithAngle = zenithAngle + noise
            zenithAngle = np.clip(zenithAngle,0,np.pi/2)
            
            nl = np.dstack((np.cos(azimuthAngle)*np.sin(zenithAngle),np.sin(azimuthAngle)*np.sin(zenithAngle),np.cos(zenithAngle)))

            # 远处无效normal设置为 0，显示时候为灰色，中间颜色
            nl[mask == 0] = 0

            nl = nl.transpose(2, 0, 1)
            nl = torch.from_numpy(nl).float()

        # NHWC -> NCHW
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()


        lb = torch.from_numpy(lb).float()
        mask = torch.from_numpy(mask).float()
        valid = torch.from_numpy(valid).float()

        rawdepth = rawdepth[np.newaxis,:,:]
        rawdepth = torch.from_numpy(rawdepth).float()

        meshdepth = meshdepth[np.newaxis,:,:]
        meshdepth = torch.from_numpy(meshdepth).float()

        # input: im, 3*h*w
        # gt: lb, h*w*3
        # mask: gt!=0,h*w
        # valid: rawdepth!=0, h*w
        # rawdepth: depth with hole, 1*h*w
        # meshdepth: depth with hole, 1*h*w
        # nl: normal with noise, 3*h*w
        if self.add_normal:
            # return im, lb, mask, valid, rawdepth, meshdepth,normal,zenithAngle,zenithAngle_original
            # 训练源码
            # return im, lb, mask, valid, rawdepth, meshdepth

            # 将深度输出换成法线
            return im, lb, mask, valid, nl, meshdepth

        else:
            # return im, lb, mask, valid, rawdepth, meshdepth,np.zeros(lb.shape),np.zeros(lb.shape),np.zeros(lb.shape)
            # 训练源码
            return im, lb, mask, valid, rawdepth, meshdepth

#%%
# Leave code for debugging purposes
if __name__ == '__main__':
    # Config your local data path
    local_path = '/home/lab/data1/chuangbin/dataSet/matterport/v1/scans'
    bs = 1 
    n = 0 
    dst = mtLoader(root=local_path,split='small_10',mono=True,add_normal=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels, masks, valids, depths, meshdepths,nl,zenithAngle,azimuthAngle = data

        imgs = imgs.numpy()
        imgs = np.transpose(imgs, [0,2,3,1])
        imgs = imgs+0.5
        # mean, median, small, mid, large = eval_normal_pixel_me(labels,normal)
        # print("Evaluation  Mean Loss: mean: %.4f, median: %.4f, 11.25: %.4f, 22.5: %.4f, 30: %.4f." % (mean, median, small, mid, large)) 
        # if mean > 5 :
        #     print("error!")
        nl = nl.numpy()
        nl = 0.5*(nl+1)

        labels = labels.numpy()
        labels = 0.5*(labels+1)

        zenithAngle = zenithAngle.numpy()
        zenithAngle = zenithAngle*(180.0/np.pi)

        azimuthAngle = azimuthAngle.numpy()
        azimuthAngle = azimuthAngle*(180.0/np.pi)

        # masks = masks.numpy()
        # masks = np.repeat(masks[:, :, :, np.newaxis], 3, axis = 3)

        # valids = valids.numpy()
        # valids = np.repeat(valids[:, :, :, np.newaxis], 3, axis = 3)

        depths = depths.numpy()
        depths = np.transpose(depths, [0,2,3,1])
        depths = np.repeat(depths, 3, axis = 3)

        meshdepths = meshdepths.numpy()
        meshdepths = np.transpose(meshdepths, [0,2,3,1])
        meshdepths = np.repeat(meshdepths, 3, axis = 3)

        f, axarr = plt.subplots(bs,4,figsize=(16, 14),squeeze=False)
        plt.subplots_adjust(wspace =0.45, hspace =0.45)#调整子图间距
        n = n+1
        for j in range(bs):
            axarr[j][0].imshow(labels[j])
            axarr[j][0].set_title("original normal")
            # misc.imsave('../result/test/'+'or_{}_in.png'.format(n),labels[j])
            axarr[j][1].imshow(nl[j])
            axarr[j][1].set_title("noisy normal")

            im3 = axarr[j][2].imshow(azimuthAngle[j])
            divider = make_axes_locatable(axarr[j][2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im3, cax=cax, orientation='vertical')
            axarr[j][2].set_title("original zenithAngle")

            im4 = axarr[j][3].imshow(zenithAngle[j])
            divider = make_axes_locatable(axarr[j][3])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im4, cax=cax, orientation='vertical')
            axarr[j][3].set_title("noisy zenithAngle")
        plt.show()
        
        # break

        pass
    print('finish !!')



# %%
