# -*- coding: UTF-8 -*-
'''
@Author: Chuangbin Chen
@Date: 2019-11-05 14:48:07
@LastEditTime : 2020-01-06 16:56:51
@LastEditors  : Do not edit
@Description: 
'''
# -*- coding: UTF-8 -*-

from PIL import Image
import scipy.misc as m
import numpy as np

def png_reader_32bit(path, img_size=(0,0)):
    # 16-bit png will be read in as a 32-bit img
    image = Image.open(path)  
    pixel = np.array(image)
    # 因为是 法线 和 深度， 所以采用间隔取值来缩放，不能使用 其他插值方法
    if img_size[0]: #nearest interpolation 最近邻插值 本来是图像是[1024,1280]，缩放成[256,320]
        step = pixel.shape[0]//img_size[0]
        pixel = pixel[0::step, :]
        pixel = pixel[:, 0::step]

    return pixel

def png_reader_uint8(path, img_size=(0,0)):
    image = Image.open(path)
    pixel = np.array(image, dtype=np.uint8)
    if img_size[0]:
        pixel = m.imresize(pixel, (img_size[0], img_size[1]))#only works for 8 bit image

    return pixel