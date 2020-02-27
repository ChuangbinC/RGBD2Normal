#!/bin/zsh
###
 # @Author: Chuangbin Chen
 # @Date: 2019-12-25 09:45:38
 # @LastEditTime : 2020-01-09 10:57:31
 # @LastEditors  : Do not edit
 # @Description: 
 ###
CUDA_VISIBLE_DEVICES=2 python -u test_RGB.py  --testset --test_dataset matterport --test_split small_10 --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/vgg_16/  --model_full_name vgg_16_matterport_l1_1_resume_RGB_best.pkl --result_path ./result/test_rgb_mp_me/ --mono_img