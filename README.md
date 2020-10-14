<!--
 * @Copyright (c) tkianai All Rights Reserved.
 * @Author         : tkianai
 * @Github         : https://github.com/tkianai
 * @Date           : 2020-04-26 13:58:01
 * @FilePath       : /ImageCls.detectron2/README.md
 * @Description    : 
 -->


# ImageClassification.detectron2

Image classification based on detectron2.

This provides a convenient way to initialize backbone in detectron2.


## Usage

- Trained with detectron2 builtin trainer

1. Use default data flow in detectron2, you only need rename `forward_d2` to `forward`, while renaming `forward` to `forward_imgnet` in `imgcls/modeling/meta_arch/clsnet.py`

2. Create your own model config

3. Run: `python train_net_builtin.py --num-gpus <gpu number> --config-file configs/<your config file>`. For example: `sh scripts/train_net_builtin.sh`


- Trained with pytorch formal imagenet trainer [**Recommend**]

1. Read carefully with some arguments in `train_net.py`
2. Run: `sh /scripts/train_net.sh`

## Useful info
- stack exchange answer explaing top k accuracy https://stats.stackexchange.com/a/331508
- From this article https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
  - Number of nodes in the output layer matches the number of labels.
  - Sigmoid activation for each node in the output layer
  - Binary cross-entropy loss function.
- [MobileNetV1](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69) is the current backbone network

![](https://miro.medium.com/max/469/1*ylHiMKAXb57bN7uDhzldlg.png)

- Backbone is the feature extractor network
- [Great stack overflow answer](https://stackoverflow.com/a/56185441/13937378) explaining the dimensions of the input tensor to Conv2D layer
- [Difference between Inception and MobileNet](https://stackoverflow.com/a/50628710/13937378)
- The 'stem' I believe is the set of operations performed before the main blocks of MobileNet
- Discussion on omitting bias term for Conv2D in large networks [here](https://stackoverflow.com/a/51988522/13937378) also since BatchNorm2d is applied after Conv2D  the channel mean will be removed
- [Why use BatchNorm2d](https://www.aiworkbox.com/lessons/batchnorm2d-how-to-use-the-batchnorm2d-module-in-pytorch)
- [Multi channel convolutions explained](https://medium.com/apache-mxnet/multi-channel-convolutions-explained-with-ms-excel-9bbf8eb77108)
- Discussion on group param in Conv2D [here](https://stackoverflow.com/a/46538480/13937378) and [here](https://mc.ai/how-groups-work-in-pytorch-convolutions/)
