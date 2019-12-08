##python myTrain.py  --backbone-name $MODEL --data-root /mnt/sdc/zhengmeisong/data/gl2ms1m_imgs/ \
  #--head-resume-root ../py-preTrain/Head_ArcFace_Epoch_5_Batch_178192_Time_2019-12-07-13-26_checkpoint.pth \
export OMP_NUM_THREADS=4
MODEL=IR_50
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL --data-root /mnt/sdc/zhengmeisong/data/gl2ms1m_imgs/ \
  --backbone-resume-root ../py-preTrain/Backbone_IR_50_Epoch_5_Batch_178192_Time_2019-12-07-13-26_checkpoint.pth \
  --emb-size 512 \
  --head-name ArcFace \
  --loss-name Focal \
  --backbone-name IR_50 \
  --batch-size 128 \
  --lr 0.01 --lr-stages 8,12,16,18 --num-epoch 24 \
  --loss-name Focal --head-name ArcFace \
  --model-root ../py-model/${MODEL}/ \
  --gpu-ids 0 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

#batch_size 40可收敛，20不收敛,40拆成2张卡不收敛
