# ALL WE NEED IS FFM 🙃

This project is my project, and it is also a study of TensorFlow, Deep Learning(CNN, RNN, LSTM, etc.) and other Machine Learning things.

The main objective of the project is to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

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



## How to Kick the Ass 👾

### Step 1: Exploratory Data Analysis(EDA)

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

#### Missing Value Imputation 缺失值处理

最简单暴力的做法当然就是直接 drop 掉那些含有缺失值的 rows。

- 针对 numerical 特征的缺失值，可以用以下方式取代：
  - 0，缺点是可能会混淆其他本来就是 0 的数值
  - -999，用某个正常情况下不会出现的数值代替，但是选得不好可能会变成异常值，要特别对待
  - Mean，平均数
  - Median，中位数，跟平均数相比，不会被异常值干扰
- 针对 categorical 特征的缺失值，可以用以下方式取代：
  - Mode，众数，最常见的值
  - 改成 "Others" 之类的值

假设你要填补 age 这个特征，然后你有其他例如 gender 这样的特征，你可以分别计算男性和女性的 age 的 mean、median 和 mode 来填补缺失值；更复杂一点的方式是，你可以把没有缺失值的数据挑出来，用它们来训练一个 regression 或 classification 模型，用这个模型来预测缺失值。

不过其实有些算法是可以容许缺失值的，这时候可以新增一个 has_missing_value 栏位（称为 NA indicator column）。

#### Outliers Detection 野点处理

发现离群值最直观的方式就是画图表，针对单一特征可以使用 box plot；两两特征则可以使用 scatter plot。

处置离群值的方式通常是直接删除或是做变换（例如 log transformation 或 binning），当然你也可以套用处理缺失值的方式。

#### Duplicate Entries Removal 异常值处理

Duplicate 或 redundant 尤其指的是那些 features 都一样，但是 target variable 却不同的数据。

#### Feature Scaling 特征缩放

- **Standardization 标准化**

原始数据集中，因为各个特征的含义和单位不同，每个特征的取值范围可能会差异很大。例如某个二元特征的范围是 0 或 1，另一个特征的范围可能是 [0, 1000000]，由于取值范围相差过大导致了模型可能会更偏向于取值范围较大的那个特征。解决的办法就是把各种不同 scale 的特征转换成同样的 scale，称为标准化或正规化。

狭义来说，标准化专门指的是通过计算 z-score，让数据的 mean 为 0、 variance 为 1。

- **Normalization 归一化**

归一化是指把每个样本缩放到单位范数（每个样本的范数为 1），适用于计算 dot product 或者两个样本之间的相似性。除了标准化、归一化之外，其他还有通过最大、最小值，把数据的范围缩放到 [0, 1] 或 [-1, 1] 的区间缩放法，不过这个方法容易受异常值的影响。

标准化是分别对单一特征进行（针对 column）；归一化是对每个 observation 进行（针对 row）。

**1. 对 SVM、logistic regression 或其他使用 squared loss function 的演算法来说，需要 standardization；**

**2. 对 Vector Space Model 来说，需要 normalization；**

**3. 对 tree-based 的算法，基本上都不需要标准化或归一化，它们对 scale 不敏感。**

#### Feature Transformation 特征变换

针对连续值特征：

- **Rounding** 

  某些精度有到小数点后第 n 位的特征，如果你其实不需要那么精确，可以考虑 `round(value * m)` 或 `round(log(value))` 这样的做法，甚至可以把 round 之后的数值当成 categorical 特征。

- **Log Transformation**

  因为 x 越大，log(x) 增长的速度就越慢，所以取 log 的意义是可以 compress 大数和 expand 小数，换句话说就是压缩 "long tail" 和展开 "head"。假设 x 原本的范围是 [100, 1000]，log(x, 10) 之后的范围就变成 [2, 3] 了。也常常使用 log(1 + x) 或 log(x / (1 - x))。

  另外一种类似的做法是 square root 平方根或 cube root 立方根（可以用在负数）。

- ​

​


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
