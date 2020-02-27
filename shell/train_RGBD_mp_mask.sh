###
 # @Author: Chuangbin Chen
 # @Date: 2020-01-10 14:40:23
 # @LastEditTime : 2020-01-10 16:25:10
 # @LastEditors  : Do not edit
 # @Description: 屏蔽mono 输入，测试是不是属于滤波
 ###
CUDA_VISIBLE_DEVICES=1,3 python -u train_RGBD_ms.py --arch_map mask --log_dir ./runs/mono_mask --writer exp1 --num_workers 16 --tfboard --arch_F fconv_ms_mask --batch_size 10 --dataset matterport --resume --resume_model_path ./checkpoint/resume_RGB/vgg_16_in_matterport_l1_2_in_RGB_best.pkl --hybrid_loss --img_rows 256 --img_cols 320 --mono_img --model_savepath ./checkpoint/FCONV_MS_MASK/
# CUDA_VISIBLE_DEVICES=1,3 python -u train_RGBD_ms.py --arch_map mask --log_dir ./runs/mono_mask --writer exp1 --num_workers 16 --tfboard --arch_F fconv_ms --batch_size 10 --dataset matterport --resume --resume_model_path ./checkpoint/resume_RGB/vgg_16_in_matterport_l1_2_in_RGB_best.pkl --hybrid_loss --img_rows 256 --img_cols 320 --mono_img --model_savepath ./checkpoint/FCONV_MS_MASK/

