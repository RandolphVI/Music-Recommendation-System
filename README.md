# ALL WE NEED IS Tree 🌲

该项目代码为本人首次参加 Kaggle 比赛的模型代码。参加的 Kaggle 比赛内容为[音乐推荐系统](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)。比赛的任务要求如下：

>  The main objective of the project is to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

在比赛结束时，我所在的队伍在 Public Leaderboard 排名为 62 名，在 Private Leaderboard 排名为 59 名（参赛队伍为 1172 支）。

![](https://farm5.staticflickr.com/4727/25262275988_b4b3986aef_o.png)

## Requirements

- Python 3.x
- Numpy
- **CatBoost**
- **XGBoost**
- **LightBGM**
- **GBDT**
- **Libffm**
- **g++ (with C++11 and OpenMP support)**

## Dataset

[Get the research data](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)

数据来源为 Kaggle 上的比赛数据集，内容为音乐推荐。

## How to Kick the Ass 👾


### Step 1 & 2: EDA & FE
Exploratory Data Analysis 以及 Feature Engineering 的工作部分在此不进行展开，如果对如何系统合理地处理数据感兴趣，可以参考我个人博客中的该篇文章[「Music Recommendation Challenge」](http://randolph.pro/2017/12/17/%E3%80%8CKaggle%E3%80%8DMusic%20Recommendation%20Challenge/)。

### Step 3: Choose the Model

#### LIBFFM + GBDT

This model is called **Field-aware Factorization Machines**. If you want to use this model, please download [LIBFFM](http://www.csie.ntu.edu.tw/~r01922136/libffm) first.

#### CatBoost


#### XGBoost

#### LightBGM




## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)

## Reference

- [Field-aware Factorization Machines for CTR Prediction](http://ntucsu.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Field-aware Factorization Machines in a Real-world Online Advertising System](https://arxiv.org/pdf/1701.04099.pdf)
