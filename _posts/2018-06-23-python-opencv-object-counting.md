---
layout: post
title: 'python+opencv建立通用运动物体计数系统'
subtitle: '通用运动物体选择，追踪，并记录物体数量。'
description: 'python+opencv 运动物体 计数'
date: 2018-06-23
categories: 技术
tags: python opencv
---
# python+opencv建立通用运动物体计数系统
通用运动物体选择，追踪，并记录物体数量。

## 概要
本项目源代码已上传github, 这里是[地址](https://github.com/JupiterD/ObjectCounting)。

### 最终效果
![image](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/8369333.jpg)

### 测试视频
此测试视频来自youtube, 如需下载请自备梯子。

你可以在这里获取[视频](https://youtu.be/wqctLW0Hb_0)。

### 计数系统流程
1. 利用背景减去法从每一帧图像中获取运动物体
2. 追踪所有的运动物体
3. 当运动物体越过计数线时将其计数
4. 将刚被计数的运动物体的图像记录下来, 并显示在视频底部

## 步骤详解
### 1. 获取运动物体
获取运动物体的常用方法之一是**背景差分法**

背景差分法的主旨非常简单, 假设你有一张图片, 将其作为背景层, 然后在这一张图片上加入一些物体, 然后将含有其他物体的图片上的所有像素减去背景层所有的像素, 非零的那一部分就是你加入的物体的所在区域。

即:

**foreground_objects = current_frame - background_layer(前景物体 = 当前帧 - 背景帧)**

![BackgroundSubtractorImage](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/4070443.jpg)

但是在通常情况下我们很难在一个视频中找到一张完整的背景图片作为背景层, 并且在实际环境中通常会有许多光线的干扰, 比如随着太阳的移动, 建筑的影子也在随之移动, 因此只保存一张静态的背景层是远远不够的, 我们需要适时调整背景图像来确保不被环境干扰, 因此我们需要在这个视频中保存一些帧数, 并试图找出大多数的背景像素。但问题是, 我们将如何得到这个背景层, 背景层的好坏将决定我们获取的运动物体的质量。

在这个项目中我选择的是**MOG2算法**, 相比MOG算法, MOG2算法抗光干扰能力比较强。而相比KNN算法, MOG2在实验时表现的更优秀, 当然本项目中的背景差分算法可以很方便的更换, 不需要修改几行代码, 只需要重新调参即可运行, 因此有兴趣的读者可以自行比较MOG, MOG2, KNN在该测试视频下的效果。

在本项目中, 为了加快程序识别速度, 我们只识别图像下半部分的图像。

原图
![原图](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/69520523.jpg)
使用MOG2算法获取的前景
![前景](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/52739086.jpg)

#### 使用MOG2算法获取前景
使用MOG2算法获取前景的代码应该是这样的:
```python
import cv2 as cv


def train_model(cap, model, model_history, learn_rate):
    """
    训练模型

    :param cap: 视频流
    :param model: 背景差分算法模型
    :param model_history: 模型比较的历史帧数
    :param learn_rate: 背景更新速度
    """
    print("开始训练模型\n模型: {}\n训练次数: {}".format(model, model_history))
    for i in range(model_history):
        retval, frame = cap.read()

        if retval:
            model.apply(frame, None, learn_rate)
        else:
            raise IOError("图像获取失败")


def background_update(frame, model, learn_rate):
    """
    利用模型更新背景

    :param frame: 新的帧
    :param model: 背景差分算法模型
    :param learn_rate: 背景更新速度
    :return: 更新后的图像
    """
    return model.apply(frame, None, learn_rate)


if __name__ == '__main__':
    video = "./video/3.mp4"

    cap = cv.VideoCapture(video)

    print("初始化物体选择模型")
    history = 500
    var_threshold = 64
    learn_rate = 0.005
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history, var_threshold, detectShadows=False)
    train_model(cap, bg_subtractor, history, learn_rate)

    split_line = 368  # 获取图像下半部分的分割线

    while True:
        retval, frame = cap.read()

        frame_temp = frame[split_line:, :]

        if not retval:
            break

        frame_mask = background_update(frame_temp, bg_subtractor, learn_rate)

        cv.imshow("frame_mask", frame_mask)

        key = cv.waitKey(10) & 0xff
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

```

#### 去噪

在大多数情况下通过MOG2算法获取的前景会有许多噪声, 我们会使用一些常用的滤波技术去除。

去噪代码应该是类似这样的:
```python
def filter_mask(frame, kernel):
    """
    将图像去噪

    :param frame: 二值化视频帧
    :param kernel: 运算核
    :return: 去噪后的二值化图像
    """
    # 开闭运算
    closing = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    expend = cv.dilate(opening, kernel, iterations=2)
    erode = cv.erode(expend, kernel)

    # 清除低于阀值噪点, 因为可能还存在灰色像素
    threshold = cv.threshold(erode, 240, 255, cv.THRESH_BINARY)[1]

    return threshold
```
去噪后的前景
![去噪后](http://jupiterd-top-image.oss-cn-hangzhou.aliyuncs.com/18-12-3/95683695.jpg)

#### 物体轮廓检测

在获取了前景图像之后, 我们将可以在获取的图像中选取我们感兴趣的部分。

在opencv中有一个标准方法cv.findContours(其详细参数请读者自行查看opencv文档)可以帮助我们从前景中获取物体。

为了方便接下去的物体追踪工作, 我们在获取物体的同时计算物体的中心, 这将有助于我们对每个物体进行追踪。

从前景中获取物体的代码应该是这样的:
```python
import cv2 as cv
from imutils.object_detection import non_max_suppression


def get_centroid(x1, y1, x2, y2):
    """
    获取物体中心

    :param x1: x轴起点
    :param y1: y轴起点
    :param x2: x轴终点
    :param y2: y轴终点
    :return: 中心坐标 (cx, xy)
    """
    return ((x2 + x1) // 2), ((y2 + y1) // 2)


def detect_object(frame, min_width=35, min_height=35):
    """
    将二值化图像中的物体挑选出来

    :param frame: 二值化图像
    :param min_width: 物体最小宽度
    :param min_height: 物体最小高度
    :return: 每个物体的矩形框左上角x1, y1坐标, 右下角x2, y2坐标和物体中心坐标cx, cy
            [(x1, y1, x2, y2), (cx, cy)]
    """
    matches = []

    # 找到物体边界矩形
    image, contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

    # 利用非极大值抑制法避免一个物体上多个矩形(误检测多次)
    rects = np.array([(x, y, x + w, y + h) for x, y, w, h in map(cv.boundingRect, contours)])
    pick = non_max_suppression(rects, overlapThresh=0.65)

    # 从每个坐标中选出符合标准大小的坐标(物体)
    for x1, y1, x2, y2 in pick:
        # 判断物体大小是否大于设定的标准
        is_valid = (x2 - x1 > min_width) and (y2 - y1 > min_height)

        # 符合标准, 将矩形坐标和物体中心坐标添加到列表中
        if is_valid:
            centroid = get_centroid(x1, y1, x2, y2)

            matches.append([(x1, y1, x2, y2), centroid])

    return matches

```

### 2. 运动物体追踪
在本项目中, 物体追踪的方式是利用了KNN算法的思想。

即将每个物体的运动轨迹进行分类, 将距离新的位置最近的上一个位置进行匹配, 即可判断新的位置是哪一个物体。

#### 建立物体类
在进行运动物体追踪的时候, 需要记录运动物体的运动轨迹是显而易见的。并且如何判断在某一帧图像中这个物体是否仍然存在也是一个需要解决的问题。

为了判断物体是否仍然存在, 我的方法是给每一帧图像设置一个标记, 如果在追踪时追踪到了一个物体, 那就给这个物体也打上这个标记, 反之, 如果这个物体在这一帧中不存在, 那么这个物体的标记将与这一帧不相等, 那么就可以将这个物体从列表中删除。

物体类代码:
```python
class CommonObject:
    """
    记录每个物体的信息
    如: 运动轨迹坐标, 有效性
    """

    def __init__(self, rect, point, flag):
        """
        :type rect: list(x1, y1, x2, y2)
        :param rect: 矩形坐标
        :type point: list(x1, y1)
        :param point: 运动轨迹坐标(目标中心)
        :type flag: int
        :param flag: 帧标记. 检测中会向实例发送一个标记, 如果标记与实例标记相同, 则表明在这一帧中存在该物体
        """
        self.rect = list(rect)
        self.points = [point]

        self._frame_flag = flag

    def update_point(self, rect, point, flag):
        """
        更新物体运动轨迹坐标

        :type rect: list(x1, y1, x2, y2)
        :param rect: 标记物体的矩形框坐标
        :type point: list(x1, y1)
        :param point: 物体坐标中心 (cx, cy)
        :type flag: int
        :param flag: 帧标记
        """
        if len(self.points) >= 10:
            del self.points[0]

        self.points.append(point)
        self.rect = list(rect)

        self._frame_flag = flag

    def is_exist(self, flag):
        """
        判断帧标记是否与物体标记相等,
        如果相等则表明这一帧中检测到了物体

        :type flag: int
        :param flag: 帧标记
        :return: 存在, 返回True; 不存在, 返回False
        """
        return self._frame_flag == flag

    @staticmethod
    def get_last_point(common_object):
        """
        获取物体实例最后一个运动坐标

        :type common_object: CommonObject
        :param common_object: CommonObject实例
        :return: 坐标
        """
        return common_object.points[-1]

    @staticmethod
    def has_new_flag(object_and_new_flag):
        """
        用于筛选物体, 将不存在的物体删去
        在筛选时应传入一个含有物体实例和帧标记的tuple

        :type object_and_new_flag: list(CommonObject, flag)
        :return: 存在时: True, 不存在时: False
        """
        object_, new_flag = object_and_new_flag
        return object_.is_exist(new_flag)

```

#### 追踪
在追踪时因为我们获取的图像是图像的下半部分, 因此我们最终获取的物体坐标需要通过计算偏移量来获取在实际图像中的坐标。并且我们需要给每一帧图像设置一个标记。

由于我们的追踪算法的思想是KNN, 因此我们需要设置一个合理的值, 避免得到的新的位置距离上一个位置太远。

因此我们的追踪模型的代码应该是这样的:
```python
from common_object import CommonObject

import numpy as np


class ObjectTrackKNN:
    """ 追踪多个运动物体 """

    def __init__(self, split_line, centroid_threshold_square):
        """
        :type split_line: int
        :param split_line: 将图像分割为两部分的线(取下半部分为识别区域)
        :type centroid_threshold_square: int
        :param centroid_threshold_square: 运动轨迹最大距离的平方
        """
        self._split_line = split_line
        self.centroid_threshold_square = centroid_threshold_square
        
        self._frame_flag = 0

    def _update_frame_flag(self):
        """更新帧标记"""
        self._frame_flag = (self._frame_flag + 1) % 10000

    def _calculate_distance_square(self, object_last_point_list, x, y):
        """
        计算新点与每个旧点的距离

        :param object_last_point_list: 所有物体上一次运动到的坐标集合
        :param x: 新的点的x坐标
        :param y: 新的点的y坐标
        :return: 距离的平方
        """
        cx_square = np.square(object_last_point_list[:, 0] - x)
        cy_square = np.square(object_last_point_list[:, 1] - y - self._split_line)
        distance_square = cx_square + cy_square

        return distance_square

    def _calculate_offset(self, y1, y2, new_cy):
        """
        计算y轴偏移坐标

        将y轴上的坐标加上偏移量, 因为在检测时只有整个图像的split_line线下的部分
        因此要加上split_line的偏移量, 才能与原图像的区域坐标对应

        :param y1: 矩形y1坐标
        :param y2: 矩形y2坐标
        :param new_cy: 矩形中心y坐标
        :return: 偏移坐标
        """
        y1 += self._split_line
        y2 += self._split_line
        new_cy += self._split_line

        return y1, y2, new_cy

    def _update_object_list(self, object_list):
        """
        删除未被新的帧标记标记过的物体对象

        :param object_list: 物体列表
        :return: 更新后的物体列表
        """
        # 清除未被标记的物体对象
        temp = filter(CommonObject.has_new_flag, zip(object_list, [self._frame_flag] * len(object_list)))
        object_list = [each[0] for each in temp]  # temp中的每个元素的格式为(object, flag), 只需获取object

        return object_list

    def get_frame_flag(self):
        """获取当前帧标记"""
        return self._frame_flag

    def object_track(self, object_list, new_matches_list):
        """
        计算物体运动轨迹, 将新的轨迹坐标添加到对应的object中, 并更新object_list列表

        :param object_list: 物体列表
        :param new_matches_list: 检测到的新的物体坐标
        :return: 物体列表
        """
        self._update_frame_flag()

        get_last_points = map(CommonObject.get_last_point, object_list)  # 获取每个物体上一次所在位置的坐标
        object_last_point_list = np.array([last_point for last_point in get_last_points])

        # 利用KNN算法的思想, 将新的坐标进行分类, 匹配最近的轨迹
        for (x1, y1, x2, y2), (new_cx, new_cy) in new_matches_list:
            # 逼近分割线, 避免识别到的矩形框变形严重(过小)
            if new_cy < 30:
                continue

            distance_square = self._calculate_distance_square(object_last_point_list, new_cx, new_cy)

            y1, y2, new_cy = self._calculate_offset(y1, y2, new_cy)

            # 判断该点是否属于已存在的物体
            is_exist = distance_square.min() < self.centroid_threshold_square
            if is_exist:
                min_distance_index = distance_square.argmin()  # 取出与新点距离平方最小的(匹配的)旧点的序号
                selected_object = object_list[min_distance_index]  # 根据匹配的旧点序号, 取出对应的物体

                selected_object.update_point((x1, y1, x2, y2), (new_cx, new_cy), self._frame_flag)  # 更新物体运动轨迹

            # 该点不属于任何已存在的物体
            else:
                # 创建新的物体并添加到object_list中
                new_object = CommonObject((x1, y1, x2, y2), (new_cx, new_cy), self._frame_flag)
                object_list.append(new_object)

        object_list = self._update_object_list(object_list)  # 去除未被标记的物体

        return object_list

```

### 3. 计数
视频中对物体计数的方法有很多, 这里使用的是当物体越过计数线时对物体计数。

因此这就需要我们对物体的运动轨迹进行判断, 进而判断物体是否越过计数线, 并且运动的方向是朝哪个方向的, 这样就可以判断出物体是进入还是出去。

并且由于物体选择和追踪时可能会出现误差, 例如一个物体已跨过计数线, 但是获取到的新的位置在计数线的另一边, 因此再次越过了计数线, 那么将会被再次计数。对于这种问题的解决方法是添加一个可以获取物体是否被计数的标记。

因此物体类将被添加如下变量和方法:
```python
class CommonObject:
    def __init__(self):
        self._has_been_counted = False  # 记录是否已被计数

    def set_counted(self):
        """设置被计数标记, 表明该实例已被计数"""
        self._has_been_counted = True

    def is_counted(self):
        """
        返回该实例是否被计数

        :return: 如果已被计数, 返回True; 否则返回False
        """
        return self._has_been_counted

```

在判断物体是否越过计数线时只需判断最后两点是否处于计数线两侧即可。

为了接下去的计数日志做准备, 我们还将记录新的被计数的物体。

因此物体计数类的代码如下:
```python
class ObjectCounting:
    """ 物体计数 """

    def __init__(self, counting_line):
        """
        :type counting_line: int
        :param counting_line: 计数线的y轴坐标
        """
        self.counting_line = counting_line

    def get_object_count(self, object_list):
        """
        获取物体个数

        :param object_list: 物体列表
        :return: 物体进出个数和被计数的物体列表
        """
        object_in = object_out = 0
        new_counting_list = []

        for each_object in object_list:
            # 运动轨迹至少为两个, 否则无法判断
            # 并且该物体还未被计数才可计数
            if not each_object.is_counted() and len(each_object.points) > 1:
                # 需要计算运动轨迹中最后两个点的向量来判断运动方向
                new_cx, new_cy = each_object.points[-1]
                last_cx, last_cy = each_object.points[-2]

                # 判断两点是否在计数线两侧, 如果在两侧则为负数(等于零也算在两侧), 否则为正数
                is_valid = (self.counting_line - new_cy) * (self.counting_line - last_cy) <= 0
                if is_valid:
                    # 如果出去则最新的点在计数线上方
                    if new_cy - last_cy <= 0:
                        object_out += 1
                    else:
                        object_in += 1

                    new_counting_list.append(each_object)
                    each_object.set_counted()

        return object_in, object_out, new_counting_list

```

### 4. 物体计数日志
为了方便观测, 我们加入了日志功能, 日志功能将在视频的下方建立一个计数栏, 每当一个新的物体被计数时, 该物体的图像将会被加载到计数栏中。

~~~
备注:由于测试视频大小为1280px, 因此为了物体图像足够清晰, 我们将计数栏中的图像个数限制在8个, 因此每个图像宽度应该为160px。
~~~

物体计数日志类代码如下:
```python
import cv2 as cv


class ObjectCountingLog:
    """ 物体计数日志 """

    def __init__(self, split_line):
        self.split_line = split_line
        self._counting_pic_list = []

    def update_counting_pic_list(self, frame, new_counting_list):
        """
        更新物体图像列表

        :param frame: 图像
        :param new_counting_list: 被新追踪到的物体列表
        """
        # 维护此列表最多只有8个物体
        if len(new_counting_list) + len(self._counting_pic_list) > 8:
            del self._counting_pic_list[:len(new_counting_list)]

        for each_object in new_counting_list:
            x1, y1, x2, y2 = each_object.rect
            object_pic = frame[y1 - self.split_line:y2 - self.split_line, x1:x2]
            object_pic = cv.resize(object_pic, (160, 160))

            self._counting_pic_list.append(object_pic)

    def get_counting_pic_list(self):
        """获取图像列表"""
        return self._counting_pic_list

```

### 5. 管道
最后一步将是非常简单的, 因为我们只需要将每一个步骤结合起来形成管道即可, 因此其代码如下:
```python
from common_object import CommonObject
import cv2 as cv


class ObjectCountingPipeline:
    """ 运动物体 检测 - 追踪 - 计数 pipeline """

    def __init__(self, object_detection_model, object_track_model, object_counting_model):
        """
        :param object_detection_model: 物体选择模型
        :param object_track_model: 物体追踪模型
        :param object_counting_model: 物体计数模型
        """
        self._object_detection_model = object_detection_model
        self._object_track_model = object_track_model
        self._object_counting_model = object_counting_model

    def detection_object(self, frame, min_width, min_height, show_mask_frame=False):
        """
        从图像中选择运动物体

        :param frame: 二值化图像
        :param min_width: 物体的最小宽度
        :param min_height: 物体的最小高度
        :param show_mask_frame: 是否显示去噪后的图像
        :return: 每个物体的矩形框左上角x1, y1坐标, 右下角x2, y2坐标和物体中心坐标cx, cy
                [(x1, y1, x2, y2), (cx, cy)]
        """
        frame_mask = self._object_detection_model.background_update(frame)  # 标记图像中的运动物体
        filter_mask = self._object_detection_model.filter_mask(frame_mask)  # 将图像去噪
        matches_list = self._object_detection_model.detect_object(filter_mask, min_width, min_height)  # 计算运动物体的坐标

        if show_mask_frame:
            cv.imshow("img", filter_mask)

        return matches_list

    def object_track(self, object_list, new_matches_list):
        """
        追踪运动目标

        :param object_list: 物体列表
        :param new_matches_list: 检测到的新的物体坐标
        :return: 更新后的物体列表
        """
        new_object_list = self._object_track_model.object_track(object_list, new_matches_list)

        return new_object_list

    def object_counting(self, object_list):
        """
        物体计数

        :param object_list: 物体列表
        :return: 新检测到并符合要求的物体个数
        """
        object_in, object_out, new_counting_list = self._object_counting_model.get_object_count(object_list)

        return object_in, object_out, new_counting_list

    def run(self, frame, min_width, min_height, object_list, counting_log=None, show_mask_frame=False):
        """
        运行计数流程

        :param frame: 图像
        :param min_width: 物体最小宽度
        :param min_height: 物体最小高度
        :param object_list: 物体列表
        :param counting_log: 物体计数日志
        :param show_mask_frame: 是否显示去噪后的图像
        :return: 追踪到的物体列表object_list, 进入的物体个数object_in, 出去的物体个数object_out
        """
        matches_list = self.detection_object(frame, min_width, min_height, show_mask_frame)

        # 如果已存在物体则开始追踪
        if object_list:
            object_list = self.object_track(object_list, matches_list)

        # 否则需添加物体
        else:
            for rect, point in matches_list:
                new_object = CommonObject(rect, point, self._object_track_model.get_frame_flag())
                object_list.append(new_object)

        object_in, object_out, new_counting_list = self.object_counting(object_list)

        if counting_log:
            counting_log.update_counting_pic_list(frame, new_counting_list)

        return object_list, object_in, object_out

```

测试程序代码如下:
```python
from object_detection import ObjectDetection
from object_track import ObjectTrackKNN
from object_counting import ObjectCounting
from object_counting_log import ObjectCountingLog
from object_counting_pipeline import ObjectCountingPipeline

import cv2 as cv
import numpy as np
from math import sqrt


def draw_frame(frame, object_list, string_and_coordinate, font, counting_log=None, split_line=None, counting_line=None):
    """
    绘制图像

    :param frame: 图像
    :param object_list: 物体列表
    :param string_and_coordinate: 显示的字符串和坐标的列表
    :param font: 字体
    :param counting_log: 物体计数日志
    :param split_line: 检测分割线的y轴坐标
    :param counting_line: 计数线的y轴坐标
    :return: frame
    """
    height, width = frame.shape[:2]
    show_frame = np.zeros((height + 160, width, 3), dtype=np.uint8)
    show_frame[:height, :] = frame

    # 绘制分割线
    if split_line:
        cv.line(show_frame, (0, split_line), (1280, split_line), (0, 255, 0), 2)

    # 绘制计数线
    if counting_line:
        cv.line(show_frame, (0, counting_line), (1280, counting_line), (255, 255, 0), 1)

    # 绘制物体追踪矩形和运动轨迹
    for each_object in object_list:
        # 绘制矩形
        x1, y1, x2, y2 = each_object.rect
        cv.rectangle(show_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        centroid_count = len(each_object.points)
        color = (0, 0, 255)
        # 存在两个以上的点, 可以绘制轨迹
        if centroid_count >= 2:
            # 被计数过的物体和未被计数过的的物体轨迹颜色不同
            if each_object.is_counted():
                color = (0, 255, 255)

            for i in range(centroid_count - 1):
                thickness = int(sqrt((i + 1) * 2.5))  # 运动轨迹线条粗细
                cv.line(show_frame, each_object.points[i], each_object.points[i + 1], color, thickness)

        # 只存在一个点, 标记中心
        else:
            cv.circle(show_frame, each_object.points[0], 1, color, 1)

    if counting_log:
        for i, object_pic in enumerate(counting_log.get_counting_pic_list()):
            show_frame[height:, (7 - i) * 160:(8 - i) * 160] = object_pic

    for string, coordinate in string_and_coordinate:
        cv.putText(show_frame, string, coordinate, font, 1, (0, 0, 0), 1)

    return show_frame


if __name__ == '__main__':
    print("打开视频")
    video = "./video/3.mp4"
    cap = cv.VideoCapture(video)

    print("初始化物体选择模型")
    history = 500
    var_threshold = 64
    learn_rate = 0.005
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history, var_threshold, detectShadows=False)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    detection_model = ObjectDetection(bg_subtractor, history, learn_rate, kernel)
    detection_model.train_model(cap)

    print("初始化物体追踪模型")
    split_line = 368
    centroid_threshold_square = 1300
    track_model = ObjectTrackKNN(split_line, centroid_threshold_square)

    print("初始化物体计数模型")
    counting_line = split_line + 50
    counting_model = ObjectCounting(counting_line)

    print("初始化物体计数日志")
    counting_log = ObjectCountingLog(split_line)

    print("初始化pipeline")
    pipeline = ObjectCountingPipeline(detection_model, track_model, counting_model)

    print("初始化视频输出器")
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output_{}_{}_{}.avi'.format(history, var_threshold, learn_rate), fourcc, 25.0, (1280, 880))

    font = cv.FONT_HERSHEY_SIMPLEX

    object_list = []  # 检测到的物体列表
    counting_pic_list = []  # 记录已计数的物体
    object_in = object_out = 0  # 物体的进出个数
    start_time = end_time = total_time = fps = 0  # 计算fps所需
    tick_frequency = cv.getTickFrequency()

    fps_string = "fps: 0"

    retval, frame = cap.read()
    while retval:
        frame_temp = frame[split_line:, :]

        start_time = cv.getTickCount()
        object_list, new_object_in, new_object_out = pipeline.run(frame_temp, 35, 35, object_list, counting_log)
        object_in += new_object_in
        object_out += new_object_out

        counting_string = "in: {}  out: {}".format(object_in, object_out)

        string_and_coordinate = [(counting_string, (40, 40)), (fps_string, (1100, 40))]

        frame = draw_frame(frame, object_list, string_and_coordinate, font, counting_log, split_line, counting_line)

        cv.imshow("video", frame)

        out.write(frame)

        key = cv.waitKey(10) & 0xff

        retval, frame = cap.read()

        fps += 1
        end_time = cv.getTickCount()

        # 一秒更新一次fps
        total_time += (end_time - start_time) / tick_frequency
        if total_time >= 1:
            fps_string = "fps: {}".format(fps)
            fps = 0
            total_time = 0

        if key == ord('q'):
            break
        elif key == ord(' '):
            cv.waitKey(0)

    cv.destroyAllWindows()
    cap.release()

```