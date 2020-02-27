import torchvision.models as models
from models.normal_estimation_net import *
from models.vgg_16_in import *
from models.unet_3_depth import *
from models.unet_3_depth_in import *
from models.fconv_fusion import *
from models.fconv_fusion_in import *
from models.fconv_ms import *
from models.fconv_ms_mask import *
from models.loss import *
from models.normal_estimation_ms import *
from models.unet_3_mask_depth import *
from models.unet_3_mask_depth_in import *
from models.unet_3_normal_sm import *
from models.unet_3_grad import *
from models.map_conv import *
# track_running_static 是跟踪 BatchNorm 的平均值和方差的控制变量，但是本程序并没有用到
def get_model(name, track_running_static):
    model = _get_model_instance(name)

    if name == 'vgg_16':
        model = model(input_channel=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'vgg_16_in': # 跟上面的区别是 这里用的是InstanceNorm2d 上面用的是 BatchNorm2d
        model = model(input_channel=3, output_channel=3, track_running_static = track_running_static) 
    elif name == 'vgg_16_in_rgbd':# same vgg_16 but input ch num is 4
        model = model(input_channel=4, output_channel=3, track_running_static = track_running_static)    
    elif name == 'unet_3':
        model = model(input_channel=1, output_channel=3, track_running_static = track_running_static)
    elif name == 'unet_3_in':# also for pure depth input
        model = model(input_channel=1, output_channel=3, track_running_static = track_running_static)
    elif name == 'unet_3_mask':
        model = model(input_channel=1, output_channel=3, track_running_static = track_running_static)
    elif name == 'unet_3_mask_in':
        model = model(input_channel=1, output_channel=3, track_running_static = track_running_static)
    elif name == 'fconv':
        model = model(input_channel1=3, input_channel2=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'fconv_in':
        model = model(input_channel1=3, input_channel2=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'fconv_ms':
        model = model(input_channel1=3, input_channel2=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'fconv_ms_mask':
        model = model(input_channel1=3, input_channel2=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'ms':
        model = model(input_channel=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'vgg_8':
        model = model(input_channel=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'unet_3_grad':# depth is 1, RGB is 3
        model = model(input_channel=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'unet_3_grad_depth':# depth is 1, RGB is 3
        model = model(input_channel=1, output_channel=3, track_running_static = track_running_static)
    elif name == 'map_conv':# valid is 2 to 1
        model = model(input_channel=2, output_channel=1, track_running_static = track_running_static)
    
    return model
# 模型选择，normal估计模型以及
def _get_model_instance(name):
    try:
        return {
            'vgg_16': vgg_16,
            'vgg_16_in': vgg_16_in, 
            'vgg_16_in_rgbd': vgg_16_in,
            'unet_3': unet_3,
            'unet_3_in': unet_3_in,
            'unet_3_mask': unet_3_mask,
            'unet_3_mask_in': unet_3_mask_in,
            'fconv': fconv,
            'fconv_in': fconv_in,
            'fconv_ms': fconv_ms,
            'ms': vgg_16_ms,
            'vgg_8': vgg_8,
            'unet_3_grad': unet_3_grad,
            'unet_3_grad_depth': unet_3_grad,
            'map_conv': map_conv,
            'fconv_ms_mask':fconv_ms_mask,
        }[name]
    except:
        print('Model {} not available'.format(name))

def get_lossfun(name, input, label, mask, train=True):
    lossfun = _get_loss_instance(name)

    #resize label, mask to input size
    if (input.size(2) != label.size(1)):
        if name == 'l1_sm':
            step = mask.size(1)//input.size(2)
            mask = mask[:, 0::step, :]
            mask = mask[:, :, 0::step]
        else:
            step = label.size(1)//input.size(2)
            label = label[:, 0::step, :, :]
            label = label[:, :, 0::step, :]
            mask = mask[:, 0::step, :]
            mask = mask[:, :, 0::step]

    loss, df = lossfun(input, label, mask, train)
    
    return loss, df

def _get_loss_instance(name):
    try:
        return {
            'cosine': cross_cosine,
            'sine': sin_cosine,
            'l1': l1norm,
            'l1gra': l1granorm,
            'l1_normgrad': l1_normgrad,
            'l1_sm': l1_sm,
            'energy': energy,
            'gradmap': gradmap,
        }[name]
    except:
        print('loss function {} not available'.format(name))
