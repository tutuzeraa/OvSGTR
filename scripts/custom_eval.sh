config=config/GroundingDINO_SwinT_OGC_pretrain.py 
data_path=./custom_data
output=./results/custom_inference
checkpoint=../checkpoints/ovd+r/vg-pretrain-coco-swint.pth


#DEBUG=2
CUDA_VISIBLE_DEVICES=0 python main.py \
  --output_dir $output \
	-c $config --data_path $data_path  \
	--eval --resume $checkpoint --dataset_file custom --save_results \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False use_test_set=True #use_gt_box=True