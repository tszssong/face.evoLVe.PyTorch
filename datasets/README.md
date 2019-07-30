## dataloader
- folder_img_iter.py  
  和torchvison自带ImageFolder功能一致，参考[trick_1](https://zhuanlan.zhihu.com/p/68191407)添加jpeg4py，测试可加速1/3（5760张图从18s->12s)  
  ubuntu按链接方法安装  
  TODO  CentOS缺少libjpeg-turbo库，待解决  
  centOs参考这里：  
  
    Download [libjpeg-turbo.repo](https://libjpeg-turbo.org/pmwiki/uploads/Downloads/libjpeg-turbo.repo) to /etc/yum.repos.d/  
    sudo yum install libjpeg-turbo-official  
    pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git  

- triplet TODO
