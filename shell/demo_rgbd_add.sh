###
 # @Author: Chuangbin Chen
 # @Date: 2020-01-11 17:40:36
 # @LastEditTime : 2020-01-11 21:05:50
 # @LastEditors  : Do not edit
 # @Description: 
 ###
#!/bin/bash
python test_RGBD_ms.py --arch_F fconv_ms --arch_map mask --no_testset --img_path ./sample_pic/save_mono/ --depth_path ./sample_pic/save_normal/ --d_scale 40000 --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS_NORMAL/  --model_full_name fconv_ms_matterport_l1_1_hybrid_resume_best.pkl --out_path ./result/demo_add_me/ --gt_path ./sample_pic/save_GT/
