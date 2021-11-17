#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

MODEL_TYPE=$1


if [ $MODEL_TYPE == "DRAN" ]
then
  echo "Using DRAN model on COCO (Karpathy split)"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 5 \
   --model logs/DRAN/model-45280.pth \
   --infos_path logs/DRAN/infos_topdown-45280.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi


if [ $MODEL_TYPE == "DRAN_f1" ]
then
  echo "Using DRAN model on COCO (Karpathy split)"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_f1/model-20000.pth \
   --infos_path logs/DRAN_f1/infos_topdown-20000.pkl\
   --only_sent_eval 0 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_layer_2" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_layer_2/model-39620.pth \
   --infos_path logs/DRAN_layer_2/infos_topdown-39620.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 2 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_layer_3" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_layer_3/model-39620.pth \
   --infos_path logs/DRAN_layer_3/infos_topdown-39620.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 3 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_layer_4" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_layer_4/model-39000.pth \
   --infos_path logs/DRAN_layer_4/infos_topdown-39000.pkl\
   --only_sent_eval 0 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "GCN_Trans_layer_5" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_layer_5/model-39620.pth \
   --infos_path logs/DRAN_layer_5/infos_topdown-39620.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 5 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "GCN_Trans_layer_6" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_layer_6/model-39620.pth \
   --infos_path logs/DRAN_layer_6/infos_topdown-39620.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 6 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_a1_gcn" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_ablation_1_trans/model-40752.pth \
   --infos_path logs/DRAN_ablation_1_trans/infos_topdown-40752.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_a2_trans" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_ablation_2_gcn/model-40752.pth \
   --infos_path logs/DRAN_ablation_2_gcn/infos_topdown-40752.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_a3_both" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_ablation_3_both/model-40752.pth \
   --infos_path logs/DRAN_ablation_3_both/infos_topdown-40752.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi

if [ $MODEL_TYPE == "DRAN_a4_cat" ]
then
  echo "Ablation Study on layers "
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 2 --language_eval 1 --beam_size 3 \
   --model logs/DRAN_ablation_4_cat/model-40752.pth \
   --infos_path logs/DRAN_ablation_4_cat/infos_topdown-40752.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi



if [ $MODEL_TYPE == "Sub_GC_Kar" ]
then
  echo "Using Sub-GC model on COCO (Karpathy split)"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 2 \
   --gpn_nms_thres 0.75 --gpn_max_subg 10 \
   --model pretrained/sub_gc_karpathy/model-60000.pth \
   --infos_path pretrained/sub_gc_karpathy/infos_topdown-60000.pkl\
   --only_sent_eval 0
fi

if [ $MODEL_TYPE == "Full_GC_Kar" ]
then
  echo "Using Full-GC model on COCO (Karpathy split)"
  python test.py --dump_images 0 --dump_json 1 --num_images -1 \
   --num_workers 6 --language_eval 1 --beam_size 3 \
   --model pretrained/full_gc/model-33000.pth \
   --infos_path pretrained/full_gc/infos_topdown-33000.pkl\
   --only_sent_eval 1 \
   --use_gpn 0 --noun_fuse 0 --pred_emb_type 2 --gcn_layers 4 --gcn_residual 1 --gcn_bn 1
fi
