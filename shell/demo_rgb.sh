###
 # @Author: Chuangbin Chen
 # @Date: 2020-01-11 17:40:36
 # @LastEditTime : 2020-01-11 23:39:08
 # @LastEditors  : Do not edit
 # @Description: 
 ###
#!/bin/bash
python test_RGB.py --arch_RGB vgg_16 --no_testset --img_path ./sample_pic/save_mono/ --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/vgg_16/  --model_full_name vgg_16_matterport_l1_1_resume_RGB_best.pkl --out_path ./result/demo_rgb_me/ --gt_path ./sample_pic/save_GT/
