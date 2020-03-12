export OMP_NUM_THREADS=4
MODEL=MobileV2
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL --data-root /ai_data/suiyifan/data/retail_product_checkout/ \
  --emb-size 512 \
  --backbone-resume-root /ai_data/zhengmeisong/preTrainModels/Backbone_MobileV2_Epoch_20_Batch_668200_Time_2019-11-30-15-49_checkpoint.pth \
  --backbone-name $MODEL \
  --batch-size 128 \
  --lr 0.01 --lr-stages 4,8,12,16,18 --num-epoch 24 \
  --loss-name Focal --head-name ArcFace \
  --model-root ../py-model/${MODEL}/ \
  --num-classes 200 \
  --gpu-ids 4 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

