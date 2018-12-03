---
layout: post
title: 'Jupyter notebook 实用技巧记录'
subtitle: 'Jupyter notebook 实用技巧记录'
description: '记录了自己在jupyter notebook中常用的技巧'
date: 2018-09-15
categories: 技术
tags: Jupyter-notebook
---
# Jupyter notebook 实用技巧记录

*此文将会持续更新......*



## 1. 绘图

~~~python3
%matplotlib inline
~~~

使用内联方式查看图片，每个cell执行完之后无需使用show()即可查看图片。



## 2. 图像支持高分屏

~~~python
%config InlineBackend.figure_format='retina'
~~~

此方法将会使绘制出来的图片支持高分屏。

~~~python
%config InlineBackend.figure_format='svg'
~~~

此方法将会使图片以矢量图的形式显示。

**注意：** 当使用矢量图显示时，matplotlib中设置的图像会略大于正常模式下显示的图像。



## 3. 打印多个变量

~~~python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
~~~



## 4. 隐藏所有Warning

~~~python
import warnings
warnings.filterwarnings("ignore")
~~~



## 5. 在函数中打印pandas表格

使用**Ipython**库中的**display**函数

~~~python
from IPython.display import display

def display_table(df):
    df.head() # 错误使用
    display(df.head()) # 正确使用
~~~



## 6. 阻止matplotlib状态输出

在使用matplotlib绘图时，matplotlib通常会打印绘图状态，为了美观我们可以使用**分号(";")**来阻止状态输出。

使用前:

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/35933731.jpg)

使用后:

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/57398525.jpg)

**注意:**

此方式只能阻止一个cell中的**最后一个绘图函数**的状态输出，当一个cell中存在多个绘图函数时，此方法将会失效。

如果需要阻止多个状态输出，可以将这些绘图函数通过一个函数进行封装，然后在此函数后面加上分号。



使用前:

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/3445237.jpg)

使用后:

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/76856691.jpg)



## 7. 显示所有变量详细信息

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/27344830.jpg)



## 8. 显示函数帮助

**方法一:**

将光标移动到函数上，使用快捷键shift + tab，在函数底部将会显示函数帮助。

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/86540245.jpg)



**方法二:**

使用ipython的**"?"**命令，使用后函数的帮助信息将会显示在浏览器底部。

![](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/29560996.jpg)

**Tip:**

点击帮助信息的右上角的一个小箭头可以在浏览器新标签页中打开帮助。

