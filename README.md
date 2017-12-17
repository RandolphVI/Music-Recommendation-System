# ALL WE NEED IS Tree 🌲

This project is my project, and it is also a study of TensorFlow, Deep Learning(CNN, RNN, LSTM, etc.) and other Machine Learning things.

The main objective of the project is to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

## Requirements

- Python 3.x
- Numpy
- **CatBoost**
- **XGBoost**
- **LightBGM**
- **H20**
- **GBDT**
- **Libffm**
- **g++ (with C++11 and OpenMP support)**

## Dataset

[Get the research data](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)

数据来源为 Kaggle 上的比赛数据集，内容为音乐推荐。

## Overview





## How to Kick the Ass 👾

### Step 1: Exploratory Data Analysis(EDA)

#### 原始特征

- 用户特征
  - `user_id`
  - `city`
  - `gender`
  - `age`
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

- 用户特征：用户本身的各种属性，例如 `user id`、`gender`（性别）、`city`（所在的城市）等
- 音乐特征：音乐本身的各种属性，例如`song id`、`name`（歌曲名）、`artist`（演唱者）、`composer`（作曲家）、`lyricist`（作词家）、`genre_ids`（音乐风格分类）等
- 交互特征：用户对音乐做出的某项行为，该行为的 aggregation 或交叉特征，例如最近听的歌曲的曲风分布或`most_like_artist_type`喜爱的歌手的类型、`listen_count`听歌的次数等
- 上下文特征：用戶对音乐做出的某项行，该行为的 metadata，例如 `registration_init_time` 注册的时间、`source_type` 使用的设备等

有些特征是在资料 EDA 阶段就可以拿到，有些特征则需要额外的步骤（例如如透过外部的 API 或者其他模型）才能取得。

#### 联想特征

- 用户特征

  - `registration_year`: 该用户的注册年份
  - `registration_month`：该用户的注册月份
  - `registration_date`：该用户的注册日
  - `expiration_year`: 该用户的退订年份
  - `expiration_month`：该用户的退订月份
  - `expiration_date`：该用户的退订日
  - `membership_days`：该用户从注册到退订的时间天数

- 音乐特征

  - `is_featured`：判断 `artist` 中是否存在 `feat.` 信息（**`.feat`** 是 **featuring** 的缩写，如果直译的话，指「以……为特色、亮点」。**`feat.`** 在歌曲里面是指专辑表演者与另外（一个或者多个）的艺人/组合的合作，也就是请人在歌曲中客串。）
  - `smaller_song`：判断 `song_length` 是否小于 `avg_song_length` （`avg_song_length` 是 **train & test** 所出现的所有 `song_length` 的平均长度） 
  - `song_lang_boolean`： 判断 `language`  是否为 **`17.0`** 或 **`45.0`**，如果是则记为 1， 否则记为 0
  - `artist_composer`：判断 `artisit` 与 `composer` 中是否出现同样的艺人，如果有则记为 1，否则记为0（间接反映艺人的有才程度）
  - `artist_composer_lyricist`：判断 `artisit` 、 `composer` 以及 `lyricist` 中是否出现同样的艺人，如果有则记为 1，否则记为0（间接反映艺人的有才程度）
  - `genre_count`： 该歌曲的 `genre_ids` 的个数
  - `artist_count`：该歌曲的 `artist` 的个数
  - `composer_count`：该歌曲的 `composer` 的个数
  - `lyricist_count`：该歌曲的 `lyricist` 的个数
  - `count_song_played`：该歌曲 `song_id` 在 **train & test** 中出现的次数
  - `count_artist_played`：该艺人 `artist` 在 **train & test** 中出现的次数
  - `count_genre_played`：该曲风 `genre_ids` 在 **train & test** 中出现的次数
  - `count_genre_like`： 该曲风 `genre_ids` 在 **train** 中 **`target=1`**（即被喜欢）的次数
  - `genre_like_ratio`：`count_genre_like` / `count_genre_played`（反映歌曲被喜爱的程度，或者说流行的程度）
  - `song_country`：根据 `isrc` 信息得到的歌曲所属的国家信息
  - `song_publisher`：根据 `isrc` 信息得到的歌曲所属的发布商信息
  - `song_year`：根据 `isrc` 信息得到的歌曲所属的发布年份信息

- 交互特征

- 上下文特征

  - `als_model_prediction`：来自 ALS 模型的预测值，该用户对某音乐的偏好程度

  - `gbdt_model_index`: 来自 GBDT 模型的 tree index，某 observation 的自动特征

#### 2.1 Missing Value Imputation 缺失值处理

最简单暴力的做法当然就是直接 drop 掉那些含有缺失值的 rows。

- 针对 numerical 特征的缺失值，可以用以下方式取代：
  - `0`，缺点是可能会混淆其他本来就是 0 的数值
  - `-999`，用某个正常情况下不会出现的数值代替，但是选得不好可能会变成异常值，要特别对待
  - Mean，平均数（例如用户年龄信息 `bd` 存在许多异常值（存在负数、零甚至超过一百），对于那些异常值可以用 `age` 的平均值来代替）
  - Median，中位数，跟平均数相比，不会被异常值干扰
- 针对 categorical 特征的缺失值，可以用以下方式取代：
  - Mode，众数，最常见的值
  - 改成 "Others" 之类的值

假设你要填补 ` age` 这个特征，然后你有其他例如 `gender` 这样的特征，你可以分别计算男性和女性的 `bd` 的 mean、median 和 mode 来填补缺失值；更复杂一点的方式是，你可以把没有缺失值的数据挑出来，用它们来训练一个 regression 或 classification 模型，用这个模型来预测缺失值。

不过其实有些算法是可以容许缺失值的，这时候可以新增一个` has_missing_value` 栏位（称为 NA indicator column）。

#### 2.2 Outliers Detection 野点处理

发现离群值最直观的方式就是画图表，针对单一特征可以使用 box plot；两两特征则可以使用 scatter plot。

处置离群值的方式通常是直接删除或是做变换（例如 log transformation 或 binning），当然你也可以套用处理缺失值的方式。

#### 2.3 Duplicate Entries Removal 异常值处理

Duplicate 或 redundant 尤其指的是那些 features 都一样，但是 target variable 却不同的数据。

#### 2.4 Feature Scaling 特征缩放

- **2.4.1 Standardization 标准化**

原始数据集中，因为各个特征的含义和单位不同，每个特征的取值范围可能会差异很大。例如某个二元特征的范围是 0 或 1，另一个特征的范围可能是 [0, 1000000]，由于取值范围相差过大导致了模型可能会更偏向于取值范围较大的那个特征。解决的办法就是把各种不同 scale 的特征转换成同样的 scale，称为标准化或正规化。

狭义来说，标准化专门指的是通过计算 z-score，让数据的 mean 为 0、 variance 为 1。

- **2.4.2 Normalization 归一化**

归一化是指把每个样本缩放到单位范数（每个样本的范数为 1），适用于计算 dot product 或者两个样本之间的相似性。除了标准化、归一化之外，其他还有通过最大、最小值，把数据的范围缩放到 [0, 1] 或 [-1, 1] 的区间缩放法，不过这个方法容易受异常值的影响。

标准化是分别对单一特征进行（针对 column）；归一化是对每个 observation 进行（针对 row）。

- **对 SVM、logistic regression 或其他使用 squared loss function 的演算法来说，需要 standardization；**
- **对 Vector Space Model 来说，需要 normalization**；
- **对 tree-based 的算法，基本上都不需要标准化或归一化，它们对 scale 不敏感。**

#### 2.5 Feature Transformation 特征变换

针对连续值特征：

- **2.5.1 Rounding** 

  某些精度有到小数点后第 n 位的特征，如果你其实不需要那么精确，可以考虑 `round(value * m)` 或 `round(log(value))` 这样的做法，甚至可以把 round 之后的数值当成 categorical 特征。

- **2.5.2 Log Transformation**

  因为 x 越大，log(x) 增长的速度就越慢，所以取 log 的意义是可以 compress 大数和 expand 小数，换句话说就是压缩 "long tail" 和展开 "head"。假设 x 原本的范围是 [100, 1000]，log(x, 10) 之后的范围就变成 [2, 3] 了。也常常使用 log(1 + x) 或 log(x / (1 - x))。

  另外一种类似的做法是 square root 平方根或 cube root 立方根（可以用在负数）。

- **2.5.3 Binarization**

  对数值型的数据设定一个 threshold，大于就赋值为 1、小于就赋值为 0。例如 `score`，如果你只关心「及格」或「不及格」，可以直接把成绩对应到 1（`score >= 60`）和 0（`score < 60`）。或是你要做啤酒销量分析，你可以新增一个 `age >= 18` 的特征来标示出已成年。

  你有一个 `color` 的 categorical 特征，如果你不在乎实际上是什么颜色的话，其实也可以改成 `has_color`。

- **2.5.4 Binning**

  也称为 bucketization。以 `age` 这样的特征为例，你可以把所有年龄拆分成 n 段，0-20 岁、20-40 岁、40-60 岁等或是 0-18 岁、18-40 岁、40-70 岁等（等距或等量），然后把个别的年龄对应到某一段，假设 26 岁是对应到第二个 bucket，那新特征的值就是 2。这种方式是人为地指定每个 bucket 的边界值，还有另外一种拆分法是根据数据的分布来拆，称为 quantization 或 quantile binning，你只需要指定 bucket 的数量即可。

  同样的概念应用到其他地方，可以把 datetime 特征拆分成上午、中午、下午和晚上；如果是 categorical 特征，则可以先 SELECT count() ... GROUP BY，然后把出现次数小于某个 threshold 的值改成 "Other" 之类的。或者是你有一个 occupation 特征，如果你其实不需要非常准确的职业资讯的话，可以把 "Web Developer"、"iOS Developer" 或 "DBA" 这些个别的资料都改成 "Software Engineer"。

  binarization 和 binning 都是对 continuous 特征做 discretization 离散化，增强模型的非线性泛化能力。

- **2.5.5 Integer Encoding**

  也称为 label encoding。把每个 category 对应到数字，一种做法是随机对应到 0, 1, 2, 3, 4 等数字；另外一种做法是依照该值出现的频率大小的顺序来给值，例如最常出现的值给 0，依序给 1, 2, 3 等等。如果是针对一些在某种程度上有次序的 categorical 特征（称为 ordinal），例如「钻石会员」「白金会员」「黄金会员」「普通会员」，直接 mapping 成数字可能没什么问题，但是如果是类似 `color` 或 `city` 这样的没有明显大小的特征的话，还是用 one-hot encoding 比较合适。不过如果用的是 tree-based 的算法就无所谓了。

  有些 categorical 特征也可能会用数字表示（例如 id），跟 continuous 特征的差别是，数值的差异或大小对 categorical 特征来说没有太大的意义。

- **2.5.6 One-hot Encoding(OHE)**

  如果某个特征有 m 种值（例如 Taipei, Beijing, Tokyo），那它 one-hot encode 之后就会变成长度为 m 的向量：

  - Taipei: [1, 0 ,0]

  - Beijing: [0, 1, 0]

  - Tokyo: [0, 0, 1]

  你也可以改用 Dummy coding，这样就只需要产生长度为 m -1 的向量：

  - Taipei: [1, 0]
  - Beijing: [0, 1]
  - Tokyo: [0, 0]

  OHE 的缺点是容易造成特征的维度大幅增加和没办法处理之前没见过的值。

- **2.5.7 Bin-counting**

  例如在 Computational Advertising 中，如果你有针对每个 user 的「广告曝光数（包含点击和未点击）」和「广告点击数」，你就可以算出每个 user 的「点击率」，然后用这个机率来表示每个 user，反之也可以对 ad id 使用类似的做法。

  ```
  ad_id   ad_views  ad_clicks  ad_ctr
  412533  18339     1355       0.074
  423334  335       12         0.036
  345664  1244      132        0.106
  349833  35387     1244       0.035
  ```

  换个思路，如果你有一个 brand 的特征，然后你可以从 user 的购买记录中找出购买 A 品牌的人，有 70% 的人会购买 B 品牌、有 40% 的人会购买 C 品牌；购买 D 品牌的人，有 10% 的人会购买 A 品牌和 E 品牌，你可以每个品牌表示成这样：

  ```
  brand  A    B    C    D    E
  A      1.0  0.7  0.4  0.0  0.0
  B      ...
  C      ...
  D      0.1  0.0  0.0  1.0  0.1
  E      ...
  ```

- **2.5.8 LabelCount Encoding**

  类似 Bin-cunting 的做法，一样是利用现有的 count 或其他统计上的资料，差别在于 LabelCount Encoding 最后用的是次序而不是数值本身。优点是对异常值不敏感。

  ```
  ad_id   ad_clicks  ad_rank
  412533  1355       1
  423334  12         4
  345664  132        3
  349833  1244       2
  ```

- **2.5.9 Count Vectorization**

  除了可以用在 text 特征之外，如果你有 comma-seperated 的 categorical 特征也可以使用这个方法。例如电影类型 `genre`，里头的值长这样 `Action,Sci-Fi,Drama`，就可以先用 `RegexTokenizer` 转成 `Array("action", "sci-fi", "drama")`，再用 `CountVectorizer` 转成 vector。

- **2.5.10 Feature Hashing**

  以 user id 为例，透过一个 hash function 把每一个 user id 映射到 `(hashed_1, hashed_2, ..., hashed_m)` 的某个值。指定 m << user id 的取值范围，所以缺点是会有 collision（如果你的 model 足够 robust，倒也是可以不管），优点是可以良好地处理之前没见过的值和罕见的值。当然不只可以 hash 单一值，也可以 hash 一个 vector。

  你可以把 feature hashing 表示为单一栏位的数值（例如 2）或是类似 one-hot encoding 那样的多栏位的 binary 表示法（例如 [0, 0, 1]）。

- **2.5.11 Category Embedding**

- **2.5.12 User Profile**

  使用用户画像来表示每个 user id，例如用户的年龄、性别、职业、收入、居住地、偏好的各种 tag 等，把每个 user 表示成一个 feature vector。除了单一维度的特征之外，也可以建立「用户听过的歌都是哪些曲风」、「用户（30 天内）浏览过的文章都是什么分类，以 TF-IDF 的方式表达。或者是把用户所有喜欢文章对应的向量的平均值作为此用户的 profile。比如某个用户经常关注与推荐系统有关的文章，那么他的 profile 中 "CB"、"CF" 和 "推荐系统" 对应的权重值就会较高。

- **2.5.13 Rare Categorical Varibales**

  先计算好每一种 category 的数量，然后把小于某个 threshold 的 category 都改成 "Others" 之类的值。或是使用 clustering 演算法来达到同样的目的。你也可以直接建立一个新的 binary feature 叫做 rare，要来标示那些相对少见的资料点。

- **2.5.14 Unseen Categorical Variables**

  当你用 training set 的资料 fit 了一个 `StringIndexer`（和 `OneHotEncoder`），把它拿去用在 test set 上时，有一定的机率你会遇到某些 categorical 特征的值只在 test set 出现，所以对只见过 training set 的 transformer 来说，这些就是所谓的 unseen values。

  对付 unseen values 通常有几种做法：

  1. 用整个 training set + test set 来编码 categorical 特征
  2. 直接舍弃含有 unseen values 的那条记录
  3. 把 unseen values 改成 "Others" 之类的已知值。

  如果采用第一种方式，一但你把这个 transformer 拿到 production 去用时，无可避免地还是会遇到 unseen values。不过通常线上的 feature engineering 会有别的做法，例如事先把 user 或 item 的各项特征都算好（定期更新或是 data 产生的时候触发），然后以 id 为 key 存进 Redis 之类的 NoSQL 里，model 要用的时候直接用 user id / item id 拿到处理好的 feature vector。

- **2.5.15 Large Categorical Variables**

  针对那种非常大的 categorical 特征（例如 id 类的特征），如果你用的是 logistic regression，其实可以硬上 one-hot encoding。不然就是利用上面提到的 feature hashing 或 bin counting 等方式；如果是 GBDT 的话，甚至可以直接用 id 硬上，只要 tree 足够多。


#### 2.6 Feature Construction 特征构建

特征构建指的是从原有的特征中，人工地创造出新的特征，通常用来解决一般的线性模型没办法学到非线性特征的问题。其中一个重点是能否通过某些办法，在特征中加入某些「额外的资讯」，虽然也得小心数据偏见的问题。

如果你有很多 user 购物的资料，除了可以 aggregate 得到 total spend 这样的 feature 之外，也可以变换一下，变成 spend in last week、spend in last month 和 spend in last year 这种可以表示「趋势」的特征。

例如：

1. `author_avg_page_view`: 该文章作者的所有文章的平均浏览数

2. `user_visited_days_since_doc_published`: 该文章发布到该用户访问经过了多少天

3. `user_history_doc_sim_categories`: 用户读过的所有文章的分类和该篇文章的分类的 TF-IDF 的相似度

4. `user_history_doc_sim_topics`: 用户读过的所有文章的内文和该篇文章的内文的 TF-IDF 的相似度

- **2.6.1 Temporal Features 时间特征**

  对于 date / time 类型的资料，除了转换成 timestamp 和取出 day、month 和 year 做成新的栏位之外，也可以对 hour 做 binning（分成上午、中午、晚上之类的）或是对 day 做 binning（分成工作日、周末）；或是想办法查出该日期当天的天气、节日或活动等讯息，例如 `is_national_holiday` 或 `has_sport_events`。

  更进一步，用 datetime 类的资料通常也可以做成 `spend_hours_last_week` 或 `spend_money_last_week` 这种可以用来表示「趋势」的特征。

- **2.6.2 Text Features 文字特征**

- **2.6.3 Spatial Features 地理特征**

- **2.6.4 Cyclical Features**

#### 2.7 Feature Interaction 特征交互

假设你有 `A` 和 `B` 两个 continuous 特征，你可以用 `A + B`、`A - B`、`A * B` 或 `A / B` 之类的方式建立新的特征。例如 `house_age_at_purchase = house_built_date - house_purchase_date` 或是 `click_through_rate = n_clicks / n_impressions`。

还有一种类似的作法叫 Polynomial Expansion 多项式展开，当 degree 为 2 时，可以把 `(x, y)` 两个特征变成 `(x, x * x, y, x * y, y * y)` 五个特征。

#### 2.8 Feature Combination 特征组合

**特征组合主要是针对 categorical 特征，特征交互则是适用于连续值特征**。但是两者的概念是差不多的，就是把两个以上的特征透过某种方式结合在一起，变成新的特征。通常用来解决一般的线性模型没办法学到非线性特征的问题。

假设有 `gender` 和 `wealth` 两个特征，分别有 2 和 3 种取值，最简单的方式就是直接 string concatenation 组合出一个新的特征 `gender_wealth`，共有 2 x 3 = 6 种取值。因为是 categorical 特征，可以直接对 `gender_wealth `使用 `StringIndexer` 和 `OneHotEncoder`。你当然也可以一起组合 continuous 和 categorical 特征，例如 `age_wealth` 这样的特征，只是 vector 里的值就不是 0 1 而是 age 本身了。

假设 C 是 categorical 特征，N 是 continuous 特征，以下有几种有意义的组合：

```
user_id  age  gender  wealth  gender_wealth  gender_wealth_ohe   age_wealth
1        56   male    rich    male_rich      [1, 0, 0, 0, 0, 0]  [56, 0, 0]
2        30   male    middle  male_middle    [0, 1, 0, 0, 0, 0]  [0, 30, 0]
3        19   female  rich    female_rich    [0, 0, 0, 1, 0, 0]  [19, 0, 0]
4        62   female  poor    female_poor    [0, 0, 0, 0, 0, 1]  [0, 0, 62]
5        78   male    poor    male_poor      [0, 0, 1, 0, 0, 0]  [0, 0, 78]
6        34   female  middle  female_middle  [0, 0, 0, 0, 1, 0]  [0, 34, 0]
```

#### 2.9 Feature Extraction 特征提取

通常就是指 dimensionality reduction。

- **Principal Component Analysis (PCA)**
- **Latent Dirichlet Allocation (LDA)**
- **Latent Semantic Analysis (LSA)**

#### 2.10 Feature Selection 特征选择

特征选择是指通过某些方法自动地从所有的特征中挑选出有用的特征。

- **Filter Method**

  采用某一种评估指标（发散性、相关性或 Information Gain 等），单独地衡量个别特征跟 target variable 之间的关系，常用的方法有 Chi Square Test（卡方检验）。这种特征选择方式没有任何模型的参与。

  以相关性来说，也不见得跟 target variable 的相关性越高就越好。

- **Wrapper Method**

  会采用某个模型来预测你的 target variable，把特征选择想成是一个组合优化的问题，想办法找出一组特征子集能够让模型的评估结果最好。缺点是太耗时间了，实际上不常用。

- **Embedded Method**

  通常会采用一个会为特征赋予 coefficients 或 importances 的演算法，例如 Logistic Regression（特别是使用 L1 penalty）或 GBDT，直接用权重或重要性对所有特征排序，然后取前 n 个作为特征子集。

#### 2.11 Feature Learning 特征学习

也称为 Representation Learning 或 Automated Feature Engineering。

- **GBDT**
- **Neural Network: Restricted Boltzmann Machines**
- **Deep Learning: Autoencoder**

### Step 3: Choose the Model

#### LIBFFM + GBDT

This model is called **Field-aware Factorization Machines**. If you want to use this model, please download [LIBFFM](http://www.csie.ntu.edu.tw/~r01922136/libffm) first.

#### CatBoost


#### XGBoost

#### LightBGM




## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)

## Reference

- [Field-aware Factorization Machines for CTR Prediction](http://ntucsu.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Field-aware Factorization Machines in a Real-world Online Advertising System](https://arxiv.org/pdf/1701.04099.pdf)
