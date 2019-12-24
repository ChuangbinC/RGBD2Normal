###
 # @Author: Chuangbin Chen
 # @Date: 2019-12-24 09:16:12
 # @LastEditTime : 2019-12-24 09:31:45
 # @LastEditors  : Do not edit
 # @Description:
 ###
#!/bin/zsh
python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --dataset matterport --test_split small_10 --testset --testset_out_path ./result/mp_10 --d_scale 40000 --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS/  --model_full_name fconv_ms_matterport_l1_2_hybrid_best.pkl 