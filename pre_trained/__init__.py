# -*- coding: UTF-8 -*-
'''
@Author: Chuangbin Chen
@Date: 2019-11-05 14:48:07
@LastEditTime: 2019-12-24 09:42:36
@LastEditors: Do not edit
@Description: 
'''

# import torchvision.models as models
from .vgg_16_load import load_vgg_16
from .vgg_8_load import load_vgg_8
# 这个函数是用来加载预训练模型的参数
def get_premodel(model, name):
    premodel = _get_premodel_instance(name)

    if name == 'vgg_16':
        model_para = premodel(model, 'scannet')
    elif name == 'vgg_16_mp':
        model_para = premodel(model, 'mp') 
    elif name == 'vgg_16_mp_in':
        model_para = premodel(model, 'mp_in')   
    elif name == 'vgg_8_mp':
        model_para = premodel(model, 'mp')    
    
    return model_para

def _get_premodel_instance(name):
    try:
        return {
            'vgg_16': load_vgg_16,
            'vgg_16_mp': load_vgg_16,
            'vgg_16_mp_in': load_vgg_16,
            'vgg_8_mp': load_vgg_8,
        }[name]
    except:
        print('Model {} not available'.format(name))