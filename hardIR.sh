#python tri_hard_train.py --data-root /data3/zhengmeisong/lmdbTrain/gl2ms1m_dl23W1f1_150WW1/ \
python tri_hard_train.py --data-root /data3/zhengmeisong/data/gl2ms1m_dl23W1f1_150WW1_img/ \
  --num-workers 0 --num-epoch 100 \
  --bag-size 72000 --batch-size 480 --test-freq 50 --save-freq 50\
  --lr 0.01 --lr-stages 200,400,1800 --margin 0.5 \
  --backbone-name IR_50 \
  --backbone-resume-root ../py-model/IR50_ms1m_epoch120.pth \
  --gpu-ids 0,1,2,3 \
  2>&1 | tee ../py-log/tri_ir_`date +'%m_%d-%H_%M'`.log  
