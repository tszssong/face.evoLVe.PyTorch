export OMP_NUM_THREADS=6
MODEL=hr40
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL \
  --data-root /data3/zhengmeisong/data/glintv2_ms1m_img/ \
  --input-size 224,224 --emb-size 512 \
  --backbone-resume-root /ai_data/zhengmeisong/openModels/hrnetv2_w40_imagenet_pretrained.pth \
  --backbone-name $MODEL \
  --disp-freq 1000 --batch-size 500 \
  --lr 0.01 --lr-stages 4,8,12,16 --num-epoch 20 \
  --loss-name Focal --head-name Combine \
  --model-root ../py-model/${MODEL}/ \
  --num-classes 143474 \
  --gpu-ids 3,2,1,0 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

