---
layout: post
title: 'Ubuntu下matplotlib seaborn中文乱码解决'
subtitle: 'matplotlib seaborn中文乱码解决'
description: '解决matplotlib和seaborn的中文乱码问题'
date: 2018-09-15
categories: 技术
tags: Ubuntu matplotlib seaborn
---
# Ubuntu下matplotlib seaborn中文乱码解决

## 1. matplotlib

首先，在python中输入

~~~python
import matplotlib
matplotlib.get_data_path()
# 输出:
# /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data
~~~

此函数打印的目录为matplotlib配置文件和字体保存处。



第一步我们将字体保存到**/usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/fonts/ttf/**目录中，这个目录是matplotlib用来保存字体的目录。

这里我使用SimHei这个字体，其他字体也同样可以使用，但是需要注意的是放入目录的字体必须是**.ttf**格式，例如ttc等格式否是不支持的。



接下来在终端中输入:

~~~shel
sudo vim /usr/local/lib/python3.6/dist-packages/matplotlib/mpl-data/matplotlibrc
~~~

matplotlibrc为matplotlib的配置文件，打开后找到**font.sans-serif**

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/99204336.jpg)

将前面的**"#"**去除，在**":"**后面加入SimHei。

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/88334617.jpg)



然后再找到**font.family**，将其修改成如图所示的样子。

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/23579134.jpg)



保存退出后，删除主目录中**.cache/matplotlib**这个目录。

~~~shell
rm -r ~/.cache/matplotlib/
~~~

随后输入以下代码测试是否成功。

~~~python
import matplotlib.pyplot as plt
plt.title("中文")
plt.show()
~~~

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/24402577.jpg)

matplotlib已经可以正常显示中文了！



## 2. seaborn

在使用中我发现尽管matplotlib可以正常显示中文，但是有时使用seaborn时依旧会产生中文乱码问题。

这是因为seaborn是基于matplotlib的，因此seaborn的设置会覆盖matplotlib的设置。

使用以下命令可以查看seaborn的配置内容。

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/85123454.jpg)

如果在**font.sans-serif**中没有看到我们设置的中文字体，此时我们需要主动设置中文字体。

~~~python
sns.set_style({'font.sans-serif':['simhei']})
~~~

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/25013492.jpg)



此时我们就可以正常显示中文了。