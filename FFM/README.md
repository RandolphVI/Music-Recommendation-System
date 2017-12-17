# Field-aware Factorization Machine for Music Recommendation Challenge

**FFM**（Field-aware Factorization Machine）最初的概念来自Yu-Chin Juan（阮毓钦，毕业于中国台湾大学）与其比赛队员，是他们借鉴了来自 Michael Jahrer 的论文中的 **field** 概念提出了 FM 的升级版模型。通过引入 **field** 的概念，FFM 把相同性质的特征归于同一个 **field**。 FFM 起初就是用于 CTR（广告点击率预测）任务的，通过分析该比赛数据特征，该比赛任务的数据特征同 CTR 任务的特点接近，因此可以套用 FFM 模型。

Ps：FFM 的作者，也就是 Yu-Chin Juan 也参加了这个比赛（成绩比我们低 🙃）。

具体的数据处理流程以及模型的训练预测均参考 [Yu-Chin Juan 之前在 Kaggle 比赛上 CTR 任务上的方法](https://github.com/guestwalk/kaggle-2014-criteo)。


## Requirement

- 64-bit Unix-like operating system (My code based on macOS High Sierra 10.13)

- Python3

- **g++ (with C++11 and OpenMP support)**


## Dataset

[Get the research data](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)

原始数据来源为 Kaggle 上的比赛数据集，但还需要将原始数据 merge 成最终的 train.csv 与 test.csv 作为下一步的输入（该模型不涉及原始数据合并代码）。


## Step-by-step

### Step 1 Libffm + GBDT

首先要先成功编译和安装  [libffm](https://github.com/guestwalk/libffm) 与 GBDT，安装的前提是需要机器上有支持 **OpenMP** 的编译器。如果你使用 OS X 的系统，需要将代码中 `model`  文件夹下 libffm 与 GBDT 模型里面的 `Makefile` 中的第一行代码都改成：

```C
CXX = g++-x
```

其中 x 取决于你新安装的 g++ 版本（我安装的是 g++ 7），因为 OS X 自带的编译器是不支持 **OpenMP**。

之后，在代码主目录下输入：

```
make -C model/gbdt
make -C model/libffm-1.13
```

如果没有出现 Error 信息，则说明已经成功编译和安装 libffm 与 GBDT。

### Step 2  

可以直接运行 `run.py` 代码，或者分别运行各流程的代码。

#### 2.1 Make the `fc.trva.top.txt` file

```
python3 utils/count.py data/train.csv > data/fc.trva.top.txt
```







Miscellaneous
=============

1. By default we use only one thread, so it may take a long time to train the
   model. If you have multi-core CPUs, you may want to set NR_THREAD in run.py
   to use more cores. 

2. Our algorithms is non-deterministic when multiple threads are used. That
   is, the results can be slightly different when you run the script two or
   more times. In our experience, the variances of logloss generally do not 
   exceed 0.0001.

3. This script generates a prediction with around 0.44510 / 0.44500 on 
   public / private leaderboards, which are slightly worse than what we had 
   during the competition. The difference is due to some minor changes.

   If you want to reproduce our best results, please do:

     $ git checkout v1.0

   For detailed instructions, please follow README in that version.


## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)

## Reference

- [Field-aware Factorization Machines for CTR Prediction](http://ntucsu.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Field-aware Factorization Machines in a Real-world Online Advertising System](https://arxiv.org/pdf/1701.04099.pdf)
