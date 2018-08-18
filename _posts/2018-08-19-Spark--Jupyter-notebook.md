---
layout: post
title: 'Ubuntu server 18.04下Spark 2.3.1 + Jupyter notebook服务器端运行部署'
subtitle: 'Spark 2.3.1 + Jupyter notebook服务器端运行部署'
description: '在服务器端部署spark2.3.1并且利用Jupyter notebook与其交互'
date: 2018-08-18
categories: 技术
tags: Ubuntu Spark Jupyter-notebook
---
# Ubuntu server 18.04下Spark 2.3.1 + Jupyter notebook服务器端运行部署

## 一、部署安装包准备

|  名称   | 版本  |                           下载链接                           |
| :-----: | :---: | :----------------------------------------------------------: |
|  Spark  | 2.3.1 | [点击下载](http://www-us.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz) |
| Jupyter | 任意  |                         使用pip安装                          |



## 二、部署Spark

### 先决条件

#### 必备软件

* Java 7 或更高版本
* Python 2.6+ 或 Python 3.4+



### 安装Spark

在本地机器上设置和运行Spark非常简单，通常遵循安装其他Hadoop生态系统的模式进行安装。

下载Spark:

~~~shell
wget http://www-us.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
~~~

安装:

~~~shell
tar -xzf spark-2.3.1-bin-hadoop2.7.tgz
sudo mv spark-2.3.1-bin-hadoop2.7 /srv/spark-2.3.1
sudo ln -s /srv/spark-2.3.1 /srv/spark
~~~



### 设置环境

编辑Bash配置文件，将Spark添加到*$PATH*中，并设置*$SPARK_HOME*变量。

~~~shell
vim ~/.bashrc
~~~

将以下内容添加到配置文件中:

~~~shell	
export SPARK_HOME=/srv/spark
export PATH=$SPARK_HOME/bin:$PATH
export PYSPARK_PYTHON=python3
~~~

然后使Bash配置文件重新生效。

运行pyspark:

~~~shell
pyspark
~~~

如果出现SPARK字符图标，则表示运行成功。



### 简化Spark提示信息

Spark通常在执行的时候通常会打印大量的INFO日志消息。因此为了降低Spark的冗长度，可以再*$SPARK_HOME/conf*中配置log4j设置。

~~~shell
sudo cp $SPARK_HOME/conf/log4j.properties.template $SPARK_HOME/conf/log4j.properties
sudo vim $SPARK_HOME/conf/log4j.properties
~~~

然后将里面的**INFO**都改为**WARN**即可。



## 三、部署服务端Jupyter notebook

Jupyter notebook是数据科学家非常喜欢使用的交互式笔记本。

pyspark可以在ipython运行，因此我们可以使用Jupyter notebook作为与pyspark交互的工具。

由于spark部署在服务器上，因此我们也需要将jupyter也安装在服务器上，然后合理配置，使其可以被远程访问。



#### 安装Jupyter

~~~shell
sudo pip3 install jupyter
~~~

如果你没有安装过pip3，请自行搜索如何安装。



安装成功后，我们将设置远程访问。

在终端输入:

~~~shell
jupyter notebook --generate-config
ipython
~~~

打开ipython后，输入:

~~~python
from notebook.auth import passwd
passwd()
~~~

然后提示输入密码，再确认密码。

如下图所示:

![](http://p88h3xolw.bkt.clouddn.com/18-8-19/49578594.jpg)

然后复制内容**sha1:.....**，然后修改默认配置文件:

找到各个项所在位置，然后修改。

~~~python
c.NotebookApp.ip='*' # 设置访问notebook的ip，*表示所有IP，这里设置ip为都可访问  
c.NotebookApp.password = u'sha1:...' # 填写刚刚生成的密文  
c.NotebookApp.open_browser = False # 禁止notebook启动时自动打开浏览器   
~~~

修改完成后，我们已经可以远程访问Jupyter notebook了,我们只需要在浏览器中输入http://服务器ip:8888，然后再打开的页面中输入密码即可访问了，但是我们还不能与pyspark交互。



![](http://p88h3xolw.bkt.clouddn.com/18-8-19/49249765.jpg)



### 利用Jupyter notebook访问pyspark

我们只要修改了pyspark的驱动即可利用Jupyter notebook访问pyspark。

~~~shell
vim ~/.bashrc
~~~

添加以下内容:

~~~shell
export PYSPARK_DRIVER_PYTHON=ipython
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
~~~

使Bash配置文件生效，然后启动pyspark，现在pyspark就可以利用Jupyter notebook访问了。
