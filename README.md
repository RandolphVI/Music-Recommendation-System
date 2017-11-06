# ALL WE NEED IS FFM 🙃

This project is my project, and it is also a study of TensorFlow, Deep Learning(CNN, RNN, LSTM, etc.).

The main objective of the project is to solve the multi-label text classification problem based on Convolutional Neural Networks. Thus, the format of the data label is like [0, 1, 0, ..., 1, 1] according to the characteristics of such problem.

## Requirements

- Python 3.x
- **Tensorflow 1.2.0 +**
- Numpy
- Gensim
- **GBDT**
- **Libffm**
- **g++ (with C++11 and OpenMP support)**

## Data

[Get the research data](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)

数据来源为 Kaggle 上的比赛数据集，内容为音乐推荐。

## Overview



## GBDT



## LIBFFM

This most important model used in this solution is called **Field-aware Factorization Machines**. If you want to use this model, please download [LIBFFM](http://www.csie.ntu.edu.tw/~r01922136/libffm) first.



## How to kick the ass 👾

### Step 1: Exploratory Data Analysis(EDA)

数据集合

#### 原始特征

- 用户特征
  - `user_id(msno)`
  - `city`
  - `gender`
  - `bd`
- 音乐特征
  - `song_id`
  - `song_length`
  - `genre_ids`
  - `language`
  - `name`
  - `artist_name`
  - `composer`
  - `lyricist`
  - `isrc`
- 交互特征
- 上下文特征
  - `registered_via`
  - `registration_init_time`
  - `exipration_date`
  - `source_system_tab`
  - `source_screen_name`
  - `source_type`

### Step 2:  Feature Engineering

Feature Engineering 是把 raw data 转换成 features 的整个过程的总称。基本上特征工程就是个手艺活，制作的好坏全凭人的功夫，往细了讲，便是创造力与经验。

以推荐系统为例，数据集中的特征可以分成以下四种：

- 用户特征：用户本身的各种属性，例如 user id、性别、所在的城市等
- 音乐特征：音乐本身的各种属性，例如 item id、歌曲名、演唱者、作曲家、作词家、音乐风格分类等
- 交互特征：用户对音乐做出的某项行为，该行为的 aggregation 或交叉特征，例如最近听的歌曲的曲风分布或喜爱的歌手的类型
- 上下文特征：用戶对音乐做出的某项行，该行为的 metadata，例如注册的时间、使用的设备等

有些特征是在资料 EDA 阶段就可以拿到，有些特征则需要额外的步骤（例如如透过外部的 API 或者其他模型）才能取得。

#### 联想特征

- 用户特征

  - `user_days_between_registration_today`：该用户的注册时间距离今天过了几天
  - `user_days_between_exipration_today`：该用户的退订时间距离今天过了几天

- 音乐特征

  - `song_id`

- 交互特征

- 上下文特征

  - `als_model_prediction`：来自 ALS 模型的预测值，该用户对某音乐的偏好程度

  - `gbdt_model_index`: 来自 GBDT 模型的 tree index，某 observation 的自动特征

#### Missing Value Imputation






数据总共有 4 列数值特征，15 列类别特征，作者首先将13列数值特征与26列类别特征(选取出现的比例非常高的一些类型作为特征, 使用one-hot进行编码)放入gbdt中训练30棵高度均为7的树，每棵树作为一个特征, 共有30个特征, 特征的值是最后显现出值的叶子节点的序号, 即这个值为0-255, 作者最后将13+26+30一共69个特征的标签经过hash处理, 然后与10^6取模做为特征的索引, 值仍用原来特征的值, 获取到10^6个one-hot编码后的稀疏特征矩阵, 放入FFM模型中进行训练








## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)

## Reference

- ​
- [Field-aware Factorization Machines for CTR Prediction](http://ntucsu.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Field-aware Factorization Machines in a Real-world Online Advertising System](https://arxiv.org/pdf/1701.04099.pdf)
