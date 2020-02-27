###
 # @Author: Chuangbin Chen
 # @Date: 2020-01-11 17:40:36
 # @LastEditTime : 2020-01-14 14:43:23
 # @LastEditors  : Do not edit
 # @Description: 单张图像测试，模型：只有normal的输入，无Mono的输入
 ###
#!/bin/bash
python test_RGBD_ms.py --arch_F fconv_ms_mask --arch_map mask --no_testset --img_path ./sample_pic/save_mono/ --depth_path ./sample_pic/save_normal/ --d_scale 40000 --img_rows 256 --img_cols 320 --model_savepath ./checkpoint/FCONV_MS_MASK/  --model_full_name fconv_ms_mask_matterport_l1_1_hybrid_resume_best.pkl --out_path ./result/demo_mask_me/ --gt_path ./sample_pic/save_GT/
