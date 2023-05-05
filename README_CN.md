# 目录

# 模型名称

Correlation Networks(CorNet)为极端多标签文本分类( XMTC )任务的网络架构。极端多标签文本分类任务的目标是从一个非常大的标签集合中，用最相关的标签子集对输入文本序列进行标记，可以在许多实际应用中找到，例如文档标注和产品标注。

CorNet通过在深度模型的预测层添加额外的CorNet模块表示不同标签之间有用的相关性信息，利用相关性知识增强原始标签预测并输出增强的标签预测。了解更多的网络细节，请参考[CorNet论文](https://dl.acm.org/doi/pdf/10.1145/3394486.3403151) 。

## 模型架构

通过为极端多标签文本分类架构--XMLCNN增加CorNet模块，获取不同标签之间的相关性，实现多标签文本分类任务性能的提升。


## 数据集

[EUR-Lex 数据集](https://drive.google.com/file/d/15WSOexahaC-5kIcraYReFXR84TSuTejc/view?usp=sharing)是有关**欧盟法律的文件**集合。它包含许多不同类型的文件，包括条约、立法、判例法和立法提案，这些文件根据**几个正交分类方案**编制索引，共3801个类别，涉及欧洲法律的不同方面。对于多标签分类问题而言，一个样本可能同时属于多个类别。

此外，还需要下载预训练的[gensim模型](https://drive.google.com/file/d/1A_jGmpsq7dVAN0-eHZ3RZaPNL-ZdViIr/view)进行数据预处理。

* 目录结构如下：

  ```
  ├── deep_data
      ├── EUR-Lex
          ├─ train_texts.txt
          ├─ train_labels.txt
          ├─ test_texts.txt
          ├─ test_labels.txt
          ├─ test.txt
          └─ train.txt
      ├── glove.840B.300d.gensim.vectors.npy
      ├── glove.840B.300d.gensim
  ```

* 执行数据预处理：

  ```
  bash ./scripts/preprocess_eurlex.sh
  ```

* 数据预处理后的目录结构：

  ```
  ├── deep_data
      ├── EUR-Lex
          ├─ train_texts.txt
          ├─ train_labels.txt
          ├─ test_texts.txt
          ├─ test_labels.txt
          ├─ test.txt
          ├─ train.txt
          ├─ vocab.npy             
          ├─ train_texts.npy
          ├─ train_labels.npy
          ├─ test_texts.npy
          ├─ test_labels.npy
          ├─ labels_binarizer
          └─ emb_init.npy
      ├── glove.840B.300d.gensim.vectors.npy
      ├── glove.840B.300d.gensim
  ```

## 环境要求

* 硬件（Ascend）
  - 使用Ascend处理器来搭建硬件环境。

- 框架
  - [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall)

- 如需查看详情，请参见如下资源
  - [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
  - [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fzh-CN%2Fmaster%2Findex.html)

* 安装依赖的环境

  ```
  pip install requirements.txt
  ```

## 快速入门

* 本地训练：

  ```
  # 单卡训练
  python train.py --run_modelarts=Flase --is_distributed=False
  ```

  ```
  # 通过shell脚本进行8卡训练
  bash ./scripts/run_distribute_train.sh	
  ```

* 本地评估：

  ```
  python eval.py
  ```

* 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://gitee.com/link?target=https%3A%2F%2Fsupport.huaweicloud.com%2Fmodelarts%2F))

  * 在 ModelArts 上使用单卡训练 

    ```
    # (1) 在网页上设置 
    # (2) 执行a或者b
    # (3) 讲预处理好的数据集并压缩为.zip上传到桶上
    # (4) 在网页上设置启动文件为 "train.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    #     在网页上设置 "run_modelarts=True"
    #     在网页上设置 "is_distributed=False"
    ```

  * 在 ModelArts 上使用多卡训练

    ```
    # (1) 在网页上设置 
    # (2) 执行a或者b
    # (3) 讲预处理好的数据集并压缩为.zip上传到桶上
    # (4) 在网页上设置启动文件为 "train.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    #     在网页上设置 "run_modelarts=True"
    #     在网页上设置 "is_distributed=True"
    ```

## 脚本说明

### 脚本和样例代码

```
├── CorNet
    ├── ascend310_infer  #用于310推理
        ├─ inc
        	├─ utils.h
        ├─ src
        	├─ main.cc
        	├─ utils.c
        ├─ build.sh
        └─ CMakeLists.txt
    ├── configure       #模型及数据集路径及参数配置
    	├─ datasets
    		├─ AmazonCat-13K.yaml
    		├─ EUR-Lex.yaml
    		└─ Wiki-500K.yaml
    	└─ models
    		└─ CorNetXMLCNN-EUR-Lex.yaml
    ├── deepxml         #网络代码部分
    	├─ _init_.py
    	├─ callback.py
    	├─ cornet.py
    	├─ data_preprocess.py  #原始数据预处理
    	├─ data_utils.py
    	├─ dataset.py
    	├─ evaluation.py
    	├─ trainonestep.py
    	└─ xmlcnn.py
    ├── scripts         #训练脚本
    	├─ preprocess_eurlex.sh
    	├─ preprocess_other.sh
    	├─ run_distribute_train.sh
    	├─ run_distribute_train_mpi.sh
    	├─ run_infer_310.sh
    	└─ run_models.sh
    ├── eval.py           #用于模型推理
    ├── evaluation.py
    ├── export.py         #由.ckpt模型导出.mindir模型
    ├── postprocess.py    #推理部分后处理，由推理结果与输入数据的标签计算推理结果
    ├── preprocess.py     #推理部分数据预处理，生成二进制数据
    ├── README.md
    ├── README_CN.md
    ├── requirements.txt
    └── train.py          #用于拉起模型训练
```

### 脚本参数

数据集配置：

```
run_modelarts：1                  # 是否云上训练

# modelarts云上参数
data_url: ""                      # S3 数据集路径
train_url: ""                     # S3 输出路径
checkpoint_url: ""                # S3 预训练模型路径
output_path: "/cache/train"       # 真实的云上机器路径，从train_url拷贝
dataset_path: "/cache/datasets/deep_data" # 真实的云上机器路径，从data_url拷贝
load_path: "/cache/model/best_38_0.5416439549567373.ckpt" #真实的云上机器路径，从checkpoint_url拷贝

# 训练参数
dynamic_pool_length: 8
bottleneck_dim: 512
num_filters: 128
dropout: 0.5
emb_trainable: False
batch_size: 32
nb_epoch: 45
swa_warmup: 10
embedding_size: 300
```

更多配置细节请参考脚本`train.py`, `eval.py`, `export.py` 和 `config/datasets/EUR-Lex.yaml`,`config/datasets/AmazonCat-13K.yaml`,`config/datasets/Wiki-500K.yaml`,`config/models/CorNetXMLCNN-EUR-Lex.yaml`。

## 训练过程

#### 单卡训练

在Ascend设备上，使用python脚本直接开始训练(单卡)

* python命令启动

  ```
  python train.py
  ```

* shell脚本启动

  ```
  bash run_models.sh
  ```

  训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果。训练过程日志：

  ```
  [I 221206 20:37:23 train:142] Model Name: CorNetXMLCNN
  [I 221206 20:37:23 train:145] Loading Training and Validation Set
  [I 221206 20:37:24 train:160] Number of Labels: 3801
  [I 221206 20:37:24 train:161] Size of Training Set: 15249
  [I 221206 20:37:24 train:162] Size of Validation Set: 200
  [I 221206 20:37:24 train:164] Training
  [I 221206 20:37:25 train:179] labels_num:3801
  [I 221206 20:37:27 train:180] dataset size: 476
  [I 221206 20:37:33 train:214] epoch size: 45
  [I 221206 20:41:27 callback:104] Early Stop at : 1 epoch.
  [I 221206 20:41:28 callback:109] Valid loss: 0.008453961461782455, P@1: 0.15, N@1: 0.15, P@5: 0.112, N@5: 0.13021373434521952
  epoch: 2 step: 476, loss is 0.006916223559528589
  epoch time: 103371.322 ms, per step time: 217.167 ms
  [I 221206 20:43:14 callback:104] Early Stop at : 2 epoch.
  [I 221206 20:43:15 callback:109] Valid loss: 0.0072003382125071114, P@1: 0.31, N@1: 0.31, P@5: 0.196, N@5: 0.23052913737708963
  epoch: 3 step: 476, loss is 0.006499198265373707
  epoch time: 106161.619 ms, per step time: 223.029 ms
  [I 221206 20:45:04 callback:104] Early Stop at : 3 epoch.
  [I 221206 20:45:05 callback:109] Valid loss: 0.006596520476575408, P@1: 0.39, N@1: 0.39, P@5: 0.261, N@5: 0.3014825186224558
  epoch: 4 step: 476, loss is 0.006494477391242981
  epoch time: 109165.576 ms, per step time: 229.339 ms
  [I 221206 20:46:57 callback:104] Early Stop at : 4 epoch.
  [I 221206 20:46:58 callback:109] Valid loss: 0.006359181965568236, P@1: 0.385, N@1: 0.385, P@5: 0.288, N@5: 0.32757135995049913
  epoch: 5 step: 476, loss is 0.005543598439544439
  epoch time: 109397.012 ms, per step time: 229.826 ms
  [I 221206 20:48:51 callback:104] Early Stop at : 5 epoch.
  [I 221206 20:48:52 callback:109] Valid loss: 0.006111334809767348, P@1: 0.455, N@1: 0.455, P@5: 0.305, N@5: 0.3545371774537934
  epoch: 6 step: 476, loss is 0.006182706914842129
  epoch time: 103439.803 ms, per step time: 217.311 ms
  [I 221206 20:50:38 callback:104] Early Stop at : 6 epoch.
  [I 221206 20:50:40 callback:109] Valid loss: 0.005783008810664926, P@1: 0.525, N@1: 0.525, P@5: 0.335, N@5: 0.39343223342062233
  epoch: 7 step: 476, loss is 0.005146141164004803
  epoch time: 107577.709 ms, per step time: 226.004 ms
  [I 221206 20:52:34 callback:104] Early Stop at : 7 epoch.
  [I 221206 20:52:35 callback:109] Valid loss: 0.005787171024296965, P@1: 0.53, N@1: 0.53, P@5: 0.341, N@5: 0.4027243185315112
  epoch: 8 step: 476, loss is 0.004496842622756958
  epoch time: 110317.411 ms, per step time: 231.759 ms
  [I 221206 20:54:28 callback:104] Early Stop at : 8 epoch.
  [I 221206 20:54:30 callback:109] Valid loss: 0.00564238615334034, P@1: 0.575, N@1: 0.575, P@5: 0.354, N@5: 0.42305001520863394
  epoch: 9 step: 476, loss is 0.00496888579800725
  epoch time: 107726.401 ms, per step time: 226.316 ms
  [I 221206 20:56:21 callback:104] Early Stop at : 9 epoch.
  [I 221206 20:56:22 callback:109] Valid loss: 0.0055118696098881105, P@1: 0.56, N@1: 0.56, P@5: 0.371, N@5: 0.43268343204278004
  epoch: 10 step: 476, loss is 0.004366626497358084
  epoch time: 110620.702 ms, per step time: 232.396 ms
  [I 221206 20:58:16 callback:104] Early Stop at : 10 epoch.
  [I 221206 20:58:17 callback:109] Valid loss: 0.005535710935613939, P@1: 0.575, N@1: 0.575, P@5: 0.39, N@5: 0.46167164914167436
  epoch: 11 step: 476, loss is 0.003847470972687006
  epoch time: 111129.374 ms, per step time: 233.465 ms
  [I 221206 21:00:12 callback:109] Valid loss: 0.005587690443332706, P@1: 0.585, N@1: 0.585, P@5: 0.375, N@5: 0.44276899229983435
  epoch: 12 step: 476, loss is 0.0037746448069810867
  epoch time: 109372.576 ms, per step time: 229.774 ms
  [I 221206 21:02:05 callback:104] Early Stop at : 12 epoch.
  [I 221206 21:02:07 callback:109] Valid loss: 0.005524085169391972, P@1: 0.575, N@1: 0.575, P@5: 0.403, N@5: 0.46454491980304186
  epoch: 13 step: 476, loss is 0.0038265245966613293
  epoch time: 104318.255 ms, per step time: 219.156 ms
  [I 221206 21:03:55 callback:104] Early Stop at : 13 epoch.
  [I 221206 21:03:56 callback:109] Valid loss: 0.005561555203582559, P@1: 0.615, N@1: 0.615, P@5: 0.394, N@5: 0.4677300461394163
  epoch: 14 step: 476, loss is 0.003664925927296281
  epoch time: 103444.132 ms, per step time: 217.320 ms
  [I 221206 21:05:44 callback:104] Early Stop at : 14 epoch.
  [I 221206 21:05:45 callback:109] Valid loss: 0.005398926391665425, P@1: 0.65, N@1: 0.65, P@5: 0.387, N@5: 0.4705191011100305
  epoch: 15 step: 476, loss is 0.003003953956067562
  epoch time: 103678.958 ms, per step time: 217.813 ms
  [I 221206 21:07:34 callback:104] Early Stop at : 15 epoch.
  [I 221206 21:07:35 callback:109] Valid loss: 0.005349576340190002, P@1: 0.67, N@1: 0.67, P@5: 0.416, N@5: 0.49099225924648165
  epoch: 16 step: 476, loss is 0.002961152233183384
  epoch time: 106058.820 ms, per step time: 222.813 ms
  [I 221206 21:09:25 callback:109] Valid loss: 0.005370713038636106, P@1: 0.61, N@1: 0.61, P@5: 0.406, N@5: 0.4728860981858341
  epoch: 17 step: 476, loss is 0.0035449203569442034
  epoch time: 111439.646 ms, per step time: 234.117 ms
  [I 221206 21:11:22 callback:109] Valid loss: 0.005476013424673251, P@1: 0.655, N@1: 0.655, P@5: 0.401, N@5: 0.48036164098059375
  epoch: 18 step: 476, loss is 0.002875108039006591
  epoch time: 105929.118 ms, per step time: 222.540 ms
  [I 221206 21:13:12 callback:109] Valid loss: 0.005415779272360461, P@1: 0.65, N@1: 0.65, P@5: 0.415, N@5: 0.4884976539974089
  epoch: 19 step: 476, loss is 0.0024718076456338167
  epoch time: 110039.614 ms, per step time: 231.176 ms
  [I 221206 21:15:08 callback:109] Valid loss: 0.005621030727135283, P@1: 0.605, N@1: 0.605, P@5: 0.391, N@5: 0.4641035250285184
  epoch: 20 step: 476, loss is 0.0028735322412103415
  epoch time: 127251.170 ms, per step time: 267.334 ms
  [I 221206 21:17:20 callback:104] Early Stop at : 20 epoch.
  [I 221206 21:17:22 callback:109] Valid loss: 0.00551488676241466, P@1: 0.655, N@1: 0.655, P@5: 0.419, N@5: 0.4938270465684303
  epoch: 21 step: 476, loss is 0.0022650915198028088
  epoch time: 104575.768 ms, per step time: 219.697 ms
  [I 221206 21:19:12 callback:109] Valid loss: 0.005314421108258622, P@1: 0.615, N@1: 0.615, P@5: 0.42, N@5: 0.4903517127854252
  epoch: 22 step: 476, loss is 0.0022358789574354887
  epoch time: 109932.397 ms, per step time: 230.950 ms
  [I 221206 21:21:09 callback:109] Valid loss: 0.005636097198086125, P@1: 0.685, N@1: 0.685, P@5: 0.396, N@5: 0.48158027419889626
  epoch: 23 step: 476, loss is 0.0022552781738340855
  epoch time: 105221.495 ms, per step time: 221.054 ms
  [I 221206 21:23:00 callback:104] Early Stop at : 23 epoch.
  [I 221206 21:23:01 callback:109] Valid loss: 0.005471063445189169, P@1: 0.69, N@1: 0.69, P@5: 0.412, N@5: 0.49680313730765163
  epoch: 24 step: 476, loss is 0.002668126253411174
  epoch time: 105754.164 ms, per step time: 222.173 ms
  [I 221206 21:24:53 callback:104] Early Stop at : 24 epoch.
  [I 221206 21:24:55 callback:109] Valid loss: 0.0053813996990876535, P@1: 0.645, N@1: 0.645, P@5: 0.431, N@5: 0.5070014835496862
  epoch: 25 step: 476, loss is 0.0021555915009230375
  epoch time: 119987.756 ms, per step time: 252.075 ms
  [I 221206 21:27:01 callback:109] Valid loss: 0.005463898381484407, P@1: 0.65, N@1: 0.65, P@5: 0.422, N@5: 0.5024238197580305
  epoch: 26 step: 476, loss is 0.002532770624384284
  epoch time: 117755.670 ms, per step time: 247.386 ms
  [I 221206 21:29:06 callback:109] Valid loss: 0.005539836761142526, P@1: 0.675, N@1: 0.675, P@5: 0.418, N@5: 0.5019667689887573
  epoch: 27 step: 476, loss is 0.002295822836458683
  epoch time: 120114.479 ms, per step time: 252.341 ms
  [I 221206 21:31:13 callback:109] Valid loss: 0.005420369029577289, P@1: 0.68, N@1: 0.68, P@5: 0.424, N@5: 0.5052049933817727
  epoch: 28 step: 476, loss is 0.0026924496050924063
  epoch time: 104656.406 ms, per step time: 219.866 ms
  [I 221206 21:33:05 callback:109] Valid loss: 0.005584699700453452, P@1: 0.645, N@1: 0.645, P@5: 0.424, N@5: 0.49771079757500003
  epoch: 29 step: 476, loss is 0.002468355465680361
  epoch time: 104156.646 ms, per step time: 218.816 ms
  [I 221206 21:35:00 callback:109] Valid loss: 0.005668599690709796, P@1: 0.665, N@1: 0.665, P@5: 0.4, N@5: 0.4834337282071125
  epoch: 30 step: 476, loss is 0.0023284591734409332
  epoch time: 107386.632 ms, per step time: 225.602 ms
  [I 221206 21:36:57 callback:109] Valid loss: 0.005631089676171541, P@1: 0.635, N@1: 0.635, P@5: 0.416, N@5: 0.4970875297695197
  epoch: 31 step: 476, loss is 0.002341817133128643
  epoch time: 112955.104 ms, per step time: 237.301 ms
  [I 221206 21:38:58 callback:104] Early Stop at : 31 epoch.
  [I 221206 21:39:08 callback:109] Valid loss: 0.005602634511888027, P@1: 0.67, N@1: 0.67, P@5: 0.43, N@5: 0.5142499345092653
  epoch: 32 step: 476, loss is 0.0018875566311180592
  epoch time: 104657.310 ms, per step time: 219.868 ms
  [I 221206 21:41:02 callback:109] Valid loss: 0.005681491856064115, P@1: 0.685, N@1: 0.685, P@5: 0.43, N@5: 0.5090191230656914
  epoch: 33 step: 476, loss is 0.0024658450856804848
  epoch time: 120152.194 ms, per step time: 252.421 ms
  [I 221206 21:43:11 callback:109] Valid loss: 0.005698913841375283, P@1: 0.65, N@1: 0.65, P@5: 0.42, N@5: 0.49628555063366764
  epoch: 34 step: 476, loss is 0.002233546692878008
  epoch time: 103811.553 ms, per step time: 218.091 ms
  [I 221206 21:45:04 callback:109] Valid loss: 0.0056835611217788285, P@1: 0.67, N@1: 0.67, P@5: 0.408, N@5: 0.4899988470680867
  epoch: 35 step: 476, loss is 0.002109581371769309
  epoch time: 104247.491 ms, per step time: 219.007 ms
  [I 221206 21:46:58 callback:109] Valid loss: 0.005525799268590552, P@1: 0.655, N@1: 0.655, P@5: 0.418, N@5: 0.49614550818082054
  epoch: 36 step: 476, loss is 0.0019530258141458035
  epoch time: 104682.965 ms, per step time: 219.922 ms
  [I 221206 21:48:53 callback:109] Valid loss: 0.00557265683476414, P@1: 0.67, N@1: 0.67, P@5: 0.428, N@5: 0.50676649606276
  epoch: 37 step: 476, loss is 0.0019405888160690665
  epoch time: 106040.690 ms, per step time: 222.775 ms
  [I 221206 21:50:49 callback:109] Valid loss: 0.005846179888716766, P@1: 0.635, N@1: 0.635, P@5: 0.423, N@5: 0.49605837449782214
  epoch: 38 step: 476, loss is 0.002065944019705057
  epoch time: 107246.229 ms, per step time: 225.307 ms
  [I 221206 21:52:47 callback:109] Valid loss: 0.005657037786607232, P@1: 0.68, N@1: 0.68, P@5: 0.422, N@5: 0.5062942011162602
  epoch: 39 step: 476, loss is 0.0024946078192442656
  epoch time: 114661.184 ms, per step time: 240.885 ms
  [I 221206 21:54:54 callback:104] Early Stop at : 39 epoch.
  [I 221206 21:54:55 callback:109] Valid loss: 0.005855009092816285, P@1: 0.67, N@1: 0.67, P@5: 0.431, N@5: 0.5156090942991055
  epoch: 40 step: 476, loss is 0.0023201294243335724
  epoch time: 103920.471 ms, per step time: 218.320 ms
  [I 221206 21:56:51 callback:109] Valid loss: 0.005729448116783585, P@1: 0.685, N@1: 0.685, P@5: 0.429, N@5: 0.5075993447995306
  epoch: 41 step: 476, loss is 0.0018735005287453532
  epoch time: 139891.162 ms, per step time: 293.889 ms
  [I 221206 21:59:23 callback:109] Valid loss: 0.005659924354404211, P@1: 0.68, N@1: 0.68, P@5: 0.417, N@5: 0.5016814068257526
  epoch: 42 step: 476, loss is 0.001665494404733181
  epoch time: 108647.016 ms, per step time: 228.250 ms
  [I 221206 22:01:24 callback:109] Valid loss: 0.00562624992536647, P@1: 0.665, N@1: 0.665, P@5: 0.421, N@5: 0.5025668098122437
  epoch: 43 step: 476, loss is 0.0020934005733579397
  epoch time: 103872.630 ms, per step time: 218.220 ms
  [I 221206 22:03:22 callback:104] Early Stop at : 43 epoch.
  [I 221206 22:03:23 callback:109] Valid loss: 0.005657566179122243, P@1: 0.7, N@1: 0.7, P@5: 0.434, N@5: 0.5175798619537828
  epoch: 44 step: 476, loss is 0.0015542855253443122
  epoch time: 104820.230 ms, per step time: 220.211 ms
  [I 221206 22:05:24 callback:109] Valid loss: 0.00554982532880136, P@1: 0.695, N@1: 0.695, P@5: 0.421, N@5: 0.5066925764528295
  epoch: 45 step: 476, loss is 0.001915604225359857
  epoch time: 114752.408 ms, per step time: 241.076 ms
  [I 221206 22:07:34 callback:109] Valid loss: 0.00561985865767513, P@1: 0.675, N@1: 0.675, P@5: 0.42, N@5: 0.49925096810208525
  Successfully Upload /home/work/user-job-dir/outputs/model/ to s3://open-data/job/liche2022120618t085669628/output/V0001/
  [I 221206 22:07:47 train:224] Finish Training
  
  ```

#### 分布式训练

在Ascend设备上，使用shell脚本执行分布式训练示例(8卡)

* shell脚本启动

  ```
  bash run_distribute_train.sh
  ```

  训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果。训练过程日志：

  ```
  
  ```

## 评估

### 评估过程

* python命令启动

  ```
  python eval.py --dataset_path= --checkpoint_path
  ```
  
  - 参数说明：
    - --dataset_path：数据集路径，如./deep_data
    - --checkpoint_path：训练阶段保存的性能较优的ckpt路径

### 评估结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
Precision@1,3,5: 
nDCG@1,3,5: 
PSPrecision@1,3,5: 
PSnDCG@1,3,5: 
```

## 导出

### 导出过程

```
#将.ckpt模型导出为mindir模型
python export.py --dataset_path= --checkpoint_path= --format=
```

- 参数说明：
  - --dataset_path：数据集路径，如./deep_data
  - --checkpoint_path：需要进行转换的ckpt路径
  - --format：需要导出的模型格式，取值可为：MINDIR、AIR，默认为MINDIR

## 推理

### 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

###  用法

#### 相关说明

- 首先要通过执行export.py导出mindir文件
- 通过preprocess.py将数据集转为二进制文件
- 执行postprocess.py将根据mindir网络输出结果进行推理，并保存评估指标等结果

执行完整的推理脚本如下：

```
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_DIR] [DEVICE_ID]
```

### 推理结果

推理结果保存在当前路径，通过cat acc.log中看到最终精度结果。

```
[I 221205 19:58:13 postprocess:85] Model Name: CorNetXMLCNN
[I 221205 19:58:13 postprocess:87] Loading Training and Validation Set
[I 221205 19:58:13 postprocess:89] Size of Test Set: 3865
[I 221205 19:58:13 postprocess:92] Size of labels_num: 3801
[I 221205 19:58:13 postprocess:100] labels_num:3801
[I 221205 19:58:15 postprocess:101] dataset size: 3865
[I 221205 19:58:15 postprocess:106] Start Preprocess
[I 221205 19:58:21 postprocess:116] Start Save Result
[I 221205 19:58:21 postprocess:131] Precision®!: 0.7723156532988357, P@3: 0.6288917636912462, P@5: 0.518188874514877
[I 221205 19:58:21 postprocess:135] nDCG@l: 0.7723156532988357, nDCG@3: 0.6659409139985449, nDCG@5: 0.6074167700029437
[I 221205 19:58:22 postprocess:139] PSPrecision@l: 0.32689525087915616, PSPrecision@3: 0.3861224761179328, PSPrecision@5: 0.41477766571708735
[I 221205 19:58:25 postprocess:143] PSPnDCG@l: 0.32689525087915616, PSPnDCG@3: 0.3707268186832284, PSPnDCG@5: 0.39002733992968247
[I 221205 19:58:25 postprocess:145] Finish Acc
```



## 性能

### 训练性能

CorNet应用于EUR-Lex训练数据集上：

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | CorNet                                                |  CorNet                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  3090                             |
| uploaded Date              | 12/04/2022                               | 12/04/2022                 |
| MindSpore Version          | 1.9                                                        | 1.3.0                                         |
| Dataset                    | EUR-Lex                                               | EUR-Lex                               |
| Training Parameters        | epoch=45,  batch_size = 32           | epoch=30,  batch_size = 32 |
| Optimizer              | Adam                                                        | Adam                                  |
| Loss Function              | BCEWithLogitsLoss                       | BCEWithLogitsLoss        |
| outputs               | logit                                                       | logit                               |
| Loss                       | 0.005727 | 0.001199 |
| Speed                      | 250ms/step（1pcs）                                          | 37 ms/step（1pcs）                    |
| Total time                 | 97mins                                                     | 16 mins                                   |
| Parameters (M)             | 1.03GB                                                   | 263.97MB                                 |

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend       |
| ------------------- | ------------ |
| Model Version       | CorNetXMLCNN |
| Resource            | Ascend 910   |
| Uploaded Date       | 12/4/2022    |
| MindSpore Version   | 1.9          |
| Dataset             | EUR-Lex      |
| batch_size          | 32           |
| outputs             | logit        |
| Accuracy（P@5）     | 60.7  %      |
| Model for inference | 1.03GB       |

## 随机情况说明



## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
