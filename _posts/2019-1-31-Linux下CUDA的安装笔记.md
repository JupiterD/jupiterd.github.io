---
layout: post
title: 'Linux下CUDA的安装笔记'
subtitle: 'Linux下CUDA的安装笔记'
description: 'Linux下CUDA的安装教程'
date: 2019-01-31
lastmod: 2019-01-31
categories: 技术
tags: CUDA 机器学习
---
# Linux下CUDA的安装笔记

为了能够加快深度学习的计算速度，我需要利用GPU来进行计算，因此我需要安装CUDA。并且为避免下次安装时重复劳动，特写此笔记。

**注意：**

**此笔记只包含基本安装步骤，如有高级需求，请自行查看官方教程。**

附上[官方教程](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)。



## 1. 环境需求

* GCC
* 与Linux内核相对于的**内核头文件和开发包**



## 2. 安装流程

1. 在命令行中输入 `lspci | grep -i nvidia` 来查看GPU型号;
2. 查看已有的GPU是否支持CUDA，附上[官方链接](https://developer.nvidia.com/cuda-gpus)，若不支持，那就不需要继续往下看了，注意，官方列表可能存在信息滞后，如果官方上不存在对应的型号，可自行搜索是否可用;
3. 在命令行中输入 `gcc --version` 来查看是否安装 gcc，若未安装，请安装；
4. 安装与Linux内核相对应的内核头文件和开发包，对于 Ubuntu 可在命令行中输入 `sudo apt-get install linux-headers-$(uname -r)` 来进行安装，其他发行版本可点[此处](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-kernel-packages)查看对应的命令；
5. 进入 **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)** 下载页;
6. 根据需求找到下载类型；
7. 下载所需文件；
8. 根据官网的提示，安装CUDA。



## 3. 安装后操作

### 3.1 强制操作

**注意：**

**请在操作之前确认 CUDA 的版本信息和Linux发行版本，如有不同，请自行修改。**



#### 3.1.2 环境设置

在 **.bashrc (或是其他配置文件，如 .zshrc)** 中输入 `export PATH=/usr/local/cuda-10.0/bin:$PATH` ，并使其生效，以便将 CUDA 加入到环境中。若使用runfile安装方法，请详见官方教程的步骤。



#### 3.2.3 POWER 9 设置

1. 在命令行中输入 `systemctl status nvidia-persistenced` 查看 NVIDIA Persistence 守护程序是否启动，若未启动，请输入 `sudo systemctl enable nvidia-persistenced` 以启动守护程序；
2. 若在 Ubuntu 下，在命令行中输入 `sudo cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d` 和 `sudo sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules` ，若是其他发行版本，请查看官方教程，或自行寻找类似文件的位置；
3. 重启系统，以初始化上述修改。



### 3.2 推荐操作

#### 3.2.1 持久化守护程序

在 **nvidia-persistenced** 未启动的情况下，输入 `sudo /usr/bin/nvidia-persistenced --verbose` ，或是通过输入 `systemctl status nvidia-persistenced` 查看 nvidia-persistenced 的启动参数，若参数中包含了 `--verbose` ，则不需要重复输入。



#### 3.2.2 验证安装

1. 确保环境设置正确；
2. 在命令行中输入 `cuda-install-samples-10.0.sh cuda-temp` 以安装测试文件，其中 *cuda-temp* 为任意目录名。
3. 在命令行中输入 `cat /proc/driver/nvidia/version` 来验证驱动程序版本，如有情况发生，请查看官方教程；
4. 进入测试目录，确保在 **NVIDIA_CUDA-10.0_Samples** 目录下，使用 `make` 命令编译测试程序，并且可以尝试`make -j*` 命令进行多线程编译（因为编译时间并不短），其中 “ * ” 为线程数量；
5. 进入 *NVIDIA_CUDA-10.0_Samples/bin/x86_64/linux/release* 目录，运行 **deviceQuery** 文件和 **bandwidthTest** 文件以验证安装，如有意外发生，请查看官方教程。