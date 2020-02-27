###
 # @Author: Chuangbin Chen
 # @Date: 2019-12-25 23:03:52
 # @LastEditTime : 2020-01-09 20:08:15
 # @LastEditors  : Do not edit
 # @Description: 
 ###
CUDA_VISIBLE_DEVICES=1,3 python -u train_RGB.py --num_workers 16 --input rgb --tfboard --arch_RGB vgg_16 --batch_size 10 --dataset matterport --state_name vgg_16_mp --pretrained  --img_rows 256 --img_cols 320 --mono_img --model_savepath ./checkpoint/vgg_16/
