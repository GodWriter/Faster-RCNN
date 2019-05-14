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



### 2019/5/14

> dataloader完善

* create_dataset()
  * 根据config.py文件中指定的tfrecord配置，生成tfrecord文件
  * add_to_tfrecord()是创建tfrecord的核心函数，**保存bbox的特别需要注意**
    * bbox是变长，在VOC2018,2019中，仅仅只有一个bbox，但是在2012中，一张图片会有多个bbox
    * 由于是变长，需要转成byte形式，此外需要额外记录它的形装，方便之后恢复
* load_dataset()
  * 加载创建好的数据集
  * parse_function是关键
    * 需要将bbox根据所记录的形状恢复，但注意**tf.decode_raw（）中的数据类型**很重要，保存的时候是什么类型，读取出来就要是什么类型，否则会报错
    * 详情可看我的博客：<https://blog.csdn.net/GodWriter/article/details/90200179>
* test_dataset()
  * 即测试能够从tfrecord中读取信息



> 运行指令

* 首先根据自己的需要修改config/config.yml文件中的配置

* 运行指令创建tf-record

  ```bash
  python main.py --module create_dataset
  ```

* 运行指令测试数据集

  ```bash
  python main.py --module test_dataset
  ```

  