---
layout: post
title: '利用 MXNet 实现softmax回归'
subtitle: '利用 MXNet 实现softmax回归'
description: '利用 MXNet 实现softmax回归 代码实现'
date: 2019-02-15
lastmod: 2019-02-15
categories: 技术
tags: 深度学习
---
# 利用 MXNet 实现softmax回归

## 1. 不利用 Gluon接口

首先利用模型，不利用 MXNet 已有的接口进行实现。

直接上代码：

~~~python
from mxnet import autograd
from mxnet import nd
from mxnet.gluon import data as gdata


def softmax(o):
    """softmax函数."""
    o_exp = o.exp()
    return o_exp / o_exp.sum(axis=1, keepdims=True)


def net(x):
    """网络模型."""
    return softmax(nd.dot(x.reshape((-1, num_pixels)), w) + b)


def cross_entropy(y_hat, y):
    """交叉熵."""
    return -nd.pick(y_hat, y).log()


def sgd(params, learning_rate, batch_size):
    """优化算法."""
    for param in params:
        param[:] = param - (learning_rate / batch_size) * param.grad


def accuracy_measure(data_iter, net):
    """测算预测精度."""
    acc_num = count = 0

    for x, y in data_iter:
        y = y.astype('float32')
        acc_num += (net(x).argmax(axis=1) == y).sum().asscalar()
        count += y.size

    return acc_num / count


def get_fashion_mnist_labels(labels):
    """获取标签所对应的类别."""
    text_labels = [
        'T恤', '裤子', '套衫', '连衣裙', '毛衣', '拖鞋', '衬衫',
        '运动鞋', '手提包', '短靴'
    ]
    return [text_labels[int(i)] for i in labels]


def get_train_test_iter(batch_size, num_workers):
    """获取训练集与测试集的迭代器."""
    # 获取mxnet自带数据集
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)

    # 获取转换器
    transformer = gdata.vision.transforms.ToTensor()

    # 生成迭代器
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    return train_iter, test_iter

# 批量大小
batch_size = 256
num_workers = 4
train_iter, test_iter = get_train_test_iter(batch_size, num_workers)

# 像素数(参数个数)
num_pixels = 28**2
# 标签个数
num_labels = 10

w = nd.random.normal(scale=0.01, shape=(num_pixels, num_labels))
# b = nd.random.normal(shape=(1, num_labels))
b = nd.zeros(num_labels)
w.attach_grad()
b.attach_grad()

# 训练周期
num_epochs = 5
learning_rate = 0.15

for i in range(num_epochs):
    # 记录一次周期内总训练损失和正确预测的个数
    train_loss_sum = train_acc_num = 0.0
    # 一次周期内总训练个数
    count = 0
    for x, y in train_iter:
        with autograd.record():
            y_hat = net(x)
            l = cross_entropy(y_hat, y)
        l.backward()
        sgd([w, b], learning_rate, batch_size)

        # 记录数据
        y = y.astype('float32')
        train_loss_sum += l.sum().asscalar()
        train_acc_num += (y_hat.argmax(axis=1) == y).sum().asscalar()
        count += y.size
    test_acc_num = accuracy_measure(test_iter, net)
    print(f"第 {i + 1} 次训练：")
    print(f"\t训练集平均损失 -> {train_loss_sum / count}")
    print(f"\t训练集平均精度 -> {round(train_acc_num / count * 100, 2)}%")
    print(f"\t测试集平均精度 -> {round(test_acc_num * 100, 2)}%")
~~~

>第 1 次训练：
>
>​	训练集平均损失 -> 0.7630617869695028
>
>​	训练集平均精度 -> 74.75%
>
>​	测试集平均精度 -> 80.97%
>
>第 2 次训练：
>
>​	训练集平均损失 -> 0.550234276898702
>
>​	训练集平均精度 -> 81.39%
>
>​	测试集平均精度 -> 83.18%
>
>第 3 次训练：
>
>​	训练集平均损失 -> 0.5144046297073365
>
>​	训练集平均精度 -> 82.47%
>
>​	测试集平均精度 -> 83.88%
>
>第 4 次训练：
>
>​	训练集平均损失 -> 0.4906385840098063
>
>​	训练集平均精度 -> 83.28%
>
>​	测试集平均精度 -> 84.07%
>
>第 5 次训练：
>
>​	训练集平均损失 -> 0.4783866967519124
>
>​	训练集平均精度 -> 83.76%
>
>​	测试集平均精度 -> 84.3%

验证实际效果代码：

~~~python
import matplotlib.pyplot as plt

# 获取10个数据
for images, labels in test_iter:
    images = images[:10]
    labels = labels[:10]
    break

labels = get_fashion_mnist_labels(labels.asnumpy())

# 预测
predict_labels = net(images).argmax(axis=1)
predict_labels = get_fashion_mnist_labels(predict_labels.asnumpy())

titles = [f"P->{prl}\nT->{l}" for prl, l in zip(predict_labels, labels)]
results = ["正确" if prl == l else "错误" for prl, l in zip(predict_labels, labels)]
f = plt.figure(figsize=(12, 2))
axs = f.subplots(1, len(images))
for image, title, r, ax in zip(images, titles, results, axs):
    ax.imshow(image.reshape(28, 28).asnumpy())
    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_xaxis().set_label_text(r, color=('black' if r == "正确" else 'red'))
    ax.get_yaxis().set_ticks([])
plt.savefig('softmax-test.svg', bbox_inches='tight')
~~~

![softmax预测结果](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/19-2-15/softmax-test.svg "利用softmax回归的预测结果")



## 2. 利用 Gluon接口

接下来利用 MXNet 接口实现线性回归。

与 *线性回归* 类似，主要的区别只有 *全连接层* 输出从 1个 变为了 10个，和 *损失函数* 从 L2Loss 变为了 SoftmaxCrossEntropyLoss（也可用缩写 SoftmaxCELoss）。

~~~python
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
from mxnet.gluon import Trainer
from mxnet import init

# 获取训练集与测试集
batch_size = 256
num_workers = 4
# get_train_test_iter函数在不利用Gluon接口的代码中
train_iter, test_iter = get_train_test_iter(batch_size, num_workers)

# 初始化神经网络模型
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal())

# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 定义优化算法
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.15})

# 开始训练
num_epochs = 5
for i in range(num_epochs):
    # 记录一次周期内总训练损失和正确预测的个数
    train_loss_sum = train_acc_num = 0.0
    # 一次周期内总训练个数
    count = 0
    for x, y in train_iter:
        with autograd.record():
            y_hat = net(x)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(batch_size)
        
        # 记录数据
        y = y.astype('float32')
        train_loss_sum += l.sum().asscalar()
        train_acc_num += (y_hat.argmax(axis=1) == y).sum().asscalar()
        count += y.size
    test_acc_num = accuracy_measure(test_iter, net)
    print(f"第 {i + 1} 次训练：")
    print(f"\t训练集平均损失 -> {train_loss_sum / count}")
    print(f"\t训练集平均精度 -> {round(train_acc_num / count * 100, 2)}%")
    print(f"\t测试集平均精度 -> {round(test_acc_num * 100, 2)}%")
~~~

>第 1 次训练：
>
>​	训练集平均损失 -> 0.7599055559158325
>
>​	训练集平均精度 -> 74.57%
>
>​	测试集平均精度 -> 81.01%
>
>第 2 次训练：
>
>​	训练集平均损失 -> 0.5557013172785441
>
>​	训练集平均精度 -> 81.16%
>
>​	测试集平均精度 -> 82.93%
>
>第 3 次训练：
>
>​	训练集平均损失 -> 0.5111433141072591
>
>​	训练集平均精度 -> 82.77%
>
>​	测试集平均精度 -> 84.25%
>
>第 4 次训练：
>
>​	训练集平均损失 -> 0.4918146755218506
>
>​	训练集平均精度 -> 83.15%
>
>​	测试集平均精度 -> 83.98%
>
>第 5 次训练：
>
>​	训练集平均损失 -> 0.47628579686482747
>
>​	训练集平均精度 -> 83.66%
>
>​	测试集平均精度 -> 84.62%