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

