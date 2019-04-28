## Faster-RCNN

### 2019/04/28

> VOC数据集加载

* dataloader.py 中 get_voc() 是出发点

* PascalVOC 类中 self. _load_gt_roidb() 得到了每个xml文件中的信息并保存到本地

* filter_roidb() 删除了没有类别的图片标签

* append_flipped_images() 将图片水平翻转后扩充数据集，提升了数据集的平移不变性

* 运行指令

  ```bash
  python main.py --module test_dataset
  ```



> 代码顺序

* 第一步，创建dataloader.py
  * 需要什么包，去创建什么包
  * 创建类Dataset()
    * 创建get_voc()函数
    * args和config有关
      * 创建config.py，调用了args中什么超参数，就创建一个parser.add_argument()
    * 创建roid列表保存xml中信息
    * isets保存了报个数据集名称，如果想多个训练集一起训练
  * 创建PascalVOC类
    * classes作为成员变量写好
    * __init__()初始化很关键
      * _image_index_file提供了train.txt地址
      * _image_file_tmpl提供了每张图片路径的模板
      * _image_anno_tmpl提供了xml文件的路径模板
    * logger功能类似于打印日志
    * results_folder用来将提取到的xml内容保存到本地，我们这里暂时不需要保存
    * 最关键的成员变量self._roidb，最关键的成员函数self._load_gt_roidb()
* 第二步，创建main.py
  * 根据条件判断调用哪个功能
  * 我们主要测试test_dataset()

