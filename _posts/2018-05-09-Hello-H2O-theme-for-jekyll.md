---
layout: post
title: 'jupiterd.top'
subtitle: '你好H2O theme for jekyll'
description: '新建博客jupiterd.top, 对H2O theme for jekyll主题的修改的说明'
date: 2018-05-09
categories: 技术
tags: jekyll 主题
---
# jupiterd.top 
首先感谢该主题的作者[kaeyleo](https://github.com/kaeyleo)
* **原版主题**及**详细文档**请查看[jekyll-theme-H2O](https://github.com/kaeyleo/jekyll-theme-H2O)
* [在线预览 Live Demo →](http://liaokeyu.com/)

这是我的个人博客[jupiterd](http://jupiterd.top/)。 

我在我的博客上修改了他的主题，但仍保留它的风格，我所做的仅仅只是根据我的需求完善他的作品。

我将在这里记录我的想法,灵感和学习记录。 
 
目前此博客处于测试状态，因此一切都是不稳定的。 
 
![readme img](http://p88h3xolw.bkt.clouddn.com/18-5-5/15425587.jpg)

## 修改的特性
详细特性请访问原版主题[jekyll-theme-H2O](https://github.com/kaeyleo/jekyll-theme-H2O)
### 删除了夜间模式
由于夜间模式实用性不大，并且如果在浏览过程中突然转变为夜间模式会影响阅读体验，因此删除了夜间模式(不是关闭，是删除)，如需夜间模式请前往原主题。
### 背景
修改后的背景:

background by [SVGBackgrounds.com](https://www.svgbackgrounds.com/)
#### 蓝色
![](http://p88h3xolw.bkt.clouddn.com/18-5-8/6978409.jpg)
#### 粉色
![](http://p88h3xolw.bkt.clouddn.com/18-5-8/61212591.jpg)
### 代码高亮
支持的语言:
* HTML
* CSS
* Sass
* Javascript
* CoffeeScript
* Java
* C-like
* Swift
* PHP
* Go
* Python

添加了:
* C
* C++
* Git
* Haskell
* Json
* Kotlin
* Matlab
* Objective-C
* SQL
* R
* YAML

### SEO优化
箭头"→"后对应的为文章中的头信息
* 标题(title)→title
* 网站作者(author)
* ~~文章描述(description)~~
* 关键字(keywords)→tags

修改了:
* 文章描述(description)→description

添加了：
* 添加了robits.txt
* 自动生成sitemap.xml

### 写一篇文章
文章一般都放在`_post`文件夹里，每篇文章的开头都需要设置这些信息:

```
---
layout: post
title: 'H2O theme for Jekyll'
subtitle: '或许是最漂亮的Jekyll主题'
description: '这里我将测试我的代码'
date: 2017-04-18
categories: 技术
cover: 'http://on2171g4d.bkt.clouddn.com/jekyll-theme-h2o-postcover.jpg'
tags: jekyll 前端开发 设计
---
```