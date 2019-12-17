export OMP_NUM_THREADS=4
MODEL=MobileV2
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL --data-root /data_luoqi/zhengmeisong/data/glintv112/ \
  --emb-size 512 \
  --backbone-name $MODEL \
  --batch-size 128 \
  --lr 0.01 --lr-stages 4,8,12,16,18 --num-epoch 24 \
  --loss-name Focal --head-name ArcFace \
  --model-root ../py-model/${MODEL}/ \
  --num-classes 144437 \
  --gpu-ids 0 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

