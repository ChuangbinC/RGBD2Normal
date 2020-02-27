###
 # @Author: Chuangbin Chen
 # @Date: 2019-12-24 09:16:12
 # @LastEditTime : 2020-01-11 16:51:39
 # @LastEditors  : Do not edit
 # @Description:
 ###
#!/bin/zsh
CUDA_VISIBLE_DEVICES=3 python -u test_RGBD_ms.py --arch_F fconv_ms --arch_map mask --dataset matterport --test_split testsmall --testset --testset_out_path ./result/mp_add_testsmall --d_scale 40000 --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS_NORMAL/  --model_full_name fconv_ms_matterport_l1_1_hybrid_resume_best.pkl --mono_img
