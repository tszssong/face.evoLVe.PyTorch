# pytorch 版人脸识别 from ZhaoJ-face.evoLVe  
#### tricks   
Softmax比其他花式loss收敛快  
lr=0.1没有warmUp的话不收敛  
lr=0.1 + warmUp最后精度比直接用lr=0.01高   
#### 结构  
![bottleneck](disp/backbone_difference.jpg)
- resnet是原始resnet网络  
- IR是insightface里的resnet
#### 训练  
    python train.py 2>&1 | tee ../py-log/`date +'%m_%d-%H_%M_%S'`.log  
- train_prefetch.py是加入[trick_2](http://zhuanlan.zhihu.com/p/68191407)cuda预取的，加速效果没有作者说的那么明显  

- tri_train.py 用triplet loss训练  
- tri_hard_train.py 加入难样本  
  ArcFace训练lr=0.1需要加warmUp,对小模型加WarmUp也难收敛，可先Softmax训几个epoch再fintune  


#### 测试  
- utils/test_ja.py 测试  

#### TODO  
  网络初始化  
