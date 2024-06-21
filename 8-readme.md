### 接口调用实例说明

在接口类文件**classify.py**提供的接口类`ViolenceClass`中，提供了以下几个函数供测试使用：

- `load_imgs_from_path`：用于从指定路径加载图片，返回图片列表。

- `transfer_imgs_to_tensor`:用于将图片列表转变为tensor。

- `classify`：调用模型，对图片进行分类处理。返回预测的分类标签值以及模型的预测分数。

- `accuracy_score`：将预测标签列表与真实标签列表作比较，计算准确率。

下面是一个调用实例**test.py**：

```py
from classify import ViolenceClass

if __name__ == "__main__":
    # 实例化一个接口类
    pred = ViolenceClass()
    # 从指定路径加载图片
    img_list = pred.load_imgs_from_path('path/to/load/img')
    # 转换图片列表到tensor格式
    imgs = pred.transfer_imgs_to_tensor(img_list)
    # 调用模型进行分类,返回预测结果以及模型预测分数。
    preds, probes = pred.classify(imgs)
    # 计算分类的正确率
    acc = pred.accuracy_score(pred.label, preds)
    # 打印正确率
    print(acc)
```

请确保：模型存放在**test.py**同级目录下，命名为**test_model.pth**。加载图片的路径可以自行指定。

特别地，如果您的测试图片保存在**classify.py**同级目录的**test**文件夹中，您就不需要自行指定图片路径并加载了。您可以直接调用接口类中的`make_predictions`函数：

```py
from classify import ViolenceClass

if __name__ == "__main__":
    pred = ViolenceClass()
    pred.make_predictions()
```

### 模型地址
* 只使用原始数据集训练得到的模型：
    * [old](https://drive.google.com/file/d/1d5EYGS8Gr-ARRdo5hU6sXZpFzHd7yyiF/view?usp=drive_link)
* 使用扩充的训练集得到的模型：
    * [new](https://drive.google.com/file/d/1h8wElR2WYpSuTcWTKuDlcFTQCQQ0awI_/view?usp=drive_link)

**NOTE:如果需要使用不同的模型，需要更改`classify.py`中的`weight_path`**