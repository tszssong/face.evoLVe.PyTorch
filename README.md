# pytorch 版人脸识别 from ZhaoJ-face.evoLVe  
    python train.py 2>&1 | tee ../py-log/`date +'%m_%d-%H_%M_%S'`.log  
- train_prefetch.py是加入[trick_2](http://zhuanlan.zhihu.com/p/68191407)cuda预取的，加速效果没有作者说的那么明显  

- tri_train.py 用triplet loss训练  
- tri_hard_train.py 加入难样本  

#### TODO  
  需要按id shuffle    
  margin自适应， 连续5个epoch 精度大于0.95后margin+1  
