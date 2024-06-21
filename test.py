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