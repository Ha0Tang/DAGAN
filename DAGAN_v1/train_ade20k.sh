#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;
python train.py --name GauGAN_DAGAN_ade --dataset_mode ade20k --dataroot ./datasets/ade20k --niter 100 --niter_decay 100 --gpu_ids 0 --checkpoints_dir ./checkpoints --batchSize 8 --save_epoch_freq 5 --save_latest_freq 1000 
# --continue_train;
