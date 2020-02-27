###
 # @Author: Chuangbin Chen
 # @Date: 2019-12-25 23:03:52
 # @LastEditTime : 2020-01-10 16:01:30
 # @LastEditors  : Do not edit
 # @Description: 
 ###
CUDA_VISIBLE_DEVICES=0,2 python -u train_RGBD_ms.py --arch_map mask --log_dir ./runs/add_normal --writer exp1 --num_workers 16 --tfboard --arch_F fconv_ms --batch_size 10 --dataset matterport --resume --resume_model_path ./checkpoint/resume_RGB/vgg_16_in_matterport_l1_2_in_RGB_best.pkl --hybrid_loss --img_rows 256 --img_cols 320 --mono_img --model_savepath ./checkpoint/FCONV_MS_NORMAL/
