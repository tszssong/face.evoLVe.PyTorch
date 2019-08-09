# face.evoLVe from ZhaoJ  
    python train.py 2>&1 | tee ../py-log/`date +'%m_%d-%H_%M_%S'`.log  
- train_prefetch.py是加入[trick_2](http://zhuanlan.zhihu.com/p/68191407)cuda预取的，加速效果没有作者说的那么明显  

- tri_train.py 用triplet loss训练  
  当前一个bag顺序取，每次迭代其实都是一个小bag_size在找triplet，需要按id shuffle