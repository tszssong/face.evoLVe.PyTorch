python tri_hard_train.py --data-root /home/zhengmeisong/ol11/data/ms1m_emore_img \
  --num-workers 16 --num-epoch 120000\
  --bag-size 1200 --batch-size 600 --test-epoch 2000 \
  --lr 0.001 --margin 0.3\
  --backbone-resume-root ../py-model/ResNet_50_Epoch_35.pth \
  --num-loaders 1 --gpu-ids 0,1,2,3,4,5,6,7 
#  2>&1 | tee ../py-log/tri_hard_`date +'%m_%d-%H_%M_%S'`.log
