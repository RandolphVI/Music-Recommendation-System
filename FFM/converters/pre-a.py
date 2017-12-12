# -*- coding:utf-8 -*-

import csv
import sys
import argparse
import data_helpers


if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
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

# These features are dense enough (they appear in the dataset more than threshold=k times), so we include them in GBDT
target_cat_feats = data_helpers.extract_topKfeature('./dac/fc.txt', topK=50)

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        # 对数值类型特征进行 dense 数据格式处理
        for j in range(1, num_count+1):
            val = row['I{0}'.format(j)]
            # 对数值型中的缺省值标注为 -10
            if val == '':
                val = -10
            feats.append('{0}'.format(val))
        if 'tr' in args['dense_path']:
            f_d.write(row['target'] + ' ' + ' '.join(feats) + '\n')
        if 'te' in args['dense_path']:
            f_d.write('0' + ' ' + ' '.join(feats) + '\n')

        cat_feats = set()
        # 对类别型特征，如果其出现在经过 GBDT 筛选后得到的显著特征当中，那么就进行 sparse 数据格式处理
        for j in range(1, cat_count+1):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        if 'tr' in args['sparse_path']:
            f_s.write(row['target'] + ' ' + ' '.join(feats) + '\n')
        if 'te' in args['sparse_path']:
            f_s.write('0' + ' ' + ' '.join(feats) + '\n')
