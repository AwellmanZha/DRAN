#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3

MODEL_TYPE=$1

if [ $MODEL_TYPE == "DRAN" ]
then
  echo "Using DRAN model on COCO (Karpathy split)"
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 41 \
   --checkpoint_path logs/DRAN
   --start_from logs/DRAN
fi

if [ $MODEL_TYPE == "DRAN_f1" ]
then
  echo "Using DRAN model on COCO (Karpathy split)"
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 200 --save_checkpoint_every 100 --max_epochs 37 \
   --checkpoint_path logs/DRAN_f1\
   --start_from logs/DRAN_f1
fi

if [ $MODEL_TYPE == "DRAN_layer_2" ]
then
  echo "Ablation Study on layers "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 2 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 35 \
   --checkpoint_path logs/DRAN_layer_2
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_layer_3" ]
then
  echo "Ablation Study on  layers "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 3 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 35 \
   --checkpoint_path logs/DRAN_layer_3
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_layer_4" ]
then
  echo "Ablation Study on  layers "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 35 \
   --checkpoint_path logs/DRAN_layer_4
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_ablation_1_trans" ]
then
  echo "Ablation Study on model variants "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 36 \
   --checkpoint_path logs/DRAN_ablation_1_trans
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_ablation_2_gcn" ]
then
  echo "Ablation Study on model variants "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 36 \
   --checkpoint_path logs/DRAN_ablation_2_gcn
#   --start_from logs/DRANs_f2
fi

if [ $MODEL_TYPE == "DRAN_ablation_3_both" ]
then
  echo "Ablation Study on model variants "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 36 \
   --checkpoint_path logs/DRAN_ablation_3_both
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_ablation_4_cat" ]
then
  echo "Ablation Study on model variants "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 36 \
   --checkpoint_path logs/DRAN_ablation_4_cat
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_layer_5" ]
then
  echo "Ablation Study on layers "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 5 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 35 \
   --checkpoint_path logs/DRAN_layer_5
#   --start_from logs/DRAN_f2
fi

if [ $MODEL_TYPE == "DRAN_layer_6" ]
then
  echo "Ablation Study on  layers "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 6 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 35 \
   --checkpoint_path logs/DRAN_layer_6
#   --start_from logs/DRAN_f2
fi


if [ $MODEL_TYPE == "DRAN_layer_2_res" ]
then
  echo "Ablation Study on  layers "
  python train.py  --id topdown --caption_model topdown --num_workers 2 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 2 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 1000 --max_epochs 35 \
   --checkpoint_path logs/DRAN_layer_2_res
#   --start_from logs/DRAN_f2
fi


if [ $MODEL_TYPE == "Sub_GC_Kar" ]
then
  echo "Using Sub-GC model on COCO (Karpathy split)"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --batch_size 64 --save_checkpoint_every 500 --max_epochs 35\
   --checkpoint_path logs/sub_gc_karpathy
fi

if [ $MODEL_TYPE == "Full_GC_Kar" ]
then
  echo "Using Full-GC model on COCO (Karpathy split)"
  python train.py  --id topdown --caption_model topdown --num_workers 6 \
   --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --learning_rate_decay_every 3 \
   --rnn_size 1000 --input_encoding_size 1000 --att_feat_size 2048 --att_hid_size 512 --val_images_use 5000 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1 \
   --batch_size 100 --save_checkpoint_every 100 --max_epochs 42 \
   --checkpoint_path logs/full_gc\
   --start_from logs/full_gc
fi