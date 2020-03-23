  #--backbone-resume-root /ai_data/zhengmeisong/preTrainModels/Backbone_MobileV2_Epoch_20_Batch_668200_Time_2019-11-30-15-49_checkpoint.pth \
export OMP_NUM_THREADS=4
MODEL=ResNet_18
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL --data-root /ai_data/zhengmeisong/data/gl2ms1m_img/ \
  --emb-size 512 \
  --backbone-name $MODEL \
  --batch-size 512 \
  --lr 0.1 --lr-stages 4,8,12,16,18 --num-epoch 24 \
  --loss-name Focal --head-name ArcFace \
  --model-root ../py-model/${MODEL}/ \
  --disp-freq 200 --num-classes 143474 \
  --gpu-ids 2 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

