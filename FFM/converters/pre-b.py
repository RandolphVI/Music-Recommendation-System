# -*- coding:utf-8 -*-

import sys
import csv
import math
import argparse
import data_helpers

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
# nr_bins 为指数级的数，如果将所有的类型特征进行 one-hot 编码之后会得到近百万的维度，对其进行 hash trick
# 具体做法是将三种特征（类别型特征（包括频繁项与非频繁项），数值型特征，GBDT 筛选得到的显著特征）与对应的特征值结合后的特征名称进行 hash
# 然后与 nr_bins = 10^6 取余，目的是进行降维
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
# threshold 为频繁特征的阈值，默认值为 10
parser.add_argument('-t', '--threshold', type=int, default=int(100))
parser.add_argument('csv_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())


num_count = 0
cat_count = 0

for i, row in enumerate(csv.DictReader(open(args['csv_path'])), start=1):
    if i == 1:
        for item in row:
            if 'I' in item:
                num_count += 1
            if 'C' in item:
                cat_count += 1
    break

print("Numerical feature: {0}; Categorical feature: {1}".format(num_count, cat_count))


def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field-1, data_helpers.hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats


def gen_feats(row):
    feats = []
    # 对数值类型特征进行处理
    for j in range(1, num_count+1):
        field = 'I{0}'.format(j)
        value = row[field]
        # 对数值型中的缺省值标注为 -10
        if value != '':
            value = float(value)
            # 如果特征值超过 2，那么对其进行求自然对数，然后平方，最后向上取整，与该特征所在列进行组合，生成特征名称
            if value > 2:
                value = int(math.log(float(value))**2)
            else:
                value = 'SP' + str(value)
        key = field + '-' + str(value)
        feats.append(key)
    # 对类别型特征进行处理
    for j in range(1, cat_count+1):
        field = 'C{0}'.format(j)
        value = row[field]
        key = field + '-' + value
        feats.append(key)
    return feats


# 对于那些出现次数超过 threshold（默认为 10） 的类别性特征标记为频繁特征
frequent_feats = data_helpers.read_freqent_feats(args['threshold'])


with open(args['out_path'], 'w') as f:
    for row, line_gbdt in zip(csv.DictReader(open(args['csv_path'])), open(args['gbdt_path'])):
        feats = []
        # 对原始数据的数值型特征与类别型特征分别进行处理
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            # 如果特征是类别型特征，并不属于频繁特征，即出现次数少于 threshold 的类别型特征，则统一都标记成 'less'
            if type == 'C' and feat not in frequent_feats:
                feat = feat.split('-')[0] + 'less'
            # 如果特征是类别型特征，并属于频繁特征，那么其中 field 需要加上数值型特征的种类个数，作为偏移，这里为 num_count
            if type == 'C':
                field += num_count
            # 如果特征是数值型特征
            feats.append((field, feat))

        # 对于 GBDT 所构造的显著特征进行处理
        for i, feat in enumerate(line_gbdt.strip().split()[1:], start=1):
            # 其中 field 需要加上数值型特征与类别型特征的种类个数和，作为偏移，这里为 num_count + cat_count
            field = i + num_count + cat_count
            feats.append((field, str(i) + ":" + feat))

        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        if 'tr' in args['out_path']:
            f.write(row['target'] + ' ' + ' '.join(feats) + '\n')
        if 'te' in args['out_path']:
            f.write('0' + ' ' + ' '.join(feats) + '\n')
