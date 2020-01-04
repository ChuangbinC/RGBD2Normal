#!/bin/zsh
###
 # @Author: Chuangbin Chen
 # @Date: 2019-12-25 09:45:38
 # @LastEditTime : 2019-12-26 01:02:50
 # @LastEditors  : Do not edit
 # @Description: 
 ###
python test_RGB.py  --testset --test_dataset matterport --test_split testsmall --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/resume_RGB/  --model_full_name vgg_16_in_matterport_l1_2_in_RGB_best.pkl --result_path ./result/test_rgb_mp/ --mono_img