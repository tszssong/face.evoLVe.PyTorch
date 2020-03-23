#python myTrainDali.py  --backbone-name $MODEL --data-root /data3/zhengmeisong/data/glintv2_ms1m_img/ \
 #--num-classes 143474 \
export OMP_NUM_THREADS=4
MODEL=RA_92
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL \
  --data-root /data3/zhengmeisong/data/msra-glintv-orig/ \
  --input-size 224,224 --emb-size 512 \
  --backbone-resume-root /ai_data/zhengmeisong/openModels/hrnetv2_w40_imagenet_pretrained.pth \
  --backbone-name $MODEL \
  --disp-freq 10 --batch-size 512 \
  --lr 0.001 --lr-stages 4,8,12,16,18 --num-epoch 24 \
  --loss-name Focal --head-name ArcFace \
  --model-root ../py-model/${MODEL}/ \
  --num-classes 86876 \
  --gpu-ids 7,4,6,5 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

