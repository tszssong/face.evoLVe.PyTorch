export OMP_NUM_THREADS=4
MODEL=dpn107
mkdir ../py-model/${MODEL}/ 
python myTrainDali.py  --backbone-name $MODEL --data-root /data3/zhengmeisong/data/glintv2_ms1m_img/ \
  --input-size 224,224 --emb-size 512 \
  --backbone-resume-root /ai_data/zhengmeisong/openModels/dpn107_extra-1ac7121e2.pth  \
  --backbone-name $MODEL \
  --disp-freq 100 --batch-size 360 \
  --lr 0.01 --lr-stages 4,8,12,16,18 --num-epoch 24 \
  --loss-name Focal --head-name ArcFace \
  --model-root ../py-model/${MODEL}/ \
  --num-classes 143474 \
  --gpu-ids 0,1,2,3,5 2>&1 | tee ../py-log/${MODEL}_`date +'%m_%d-%H_%M'`.log

