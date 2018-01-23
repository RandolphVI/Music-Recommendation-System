# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import csv
import collections
import hashlib


data_path = './data/'

train_file = data_path + 'train.csv'
fc_trva_file = data_path + 'fc.trva.txt'
fc_file = data_path + 'fc.txt'


def count(input_file, output_file):
    num_count = 0
    cat_count = 0

    for i, row in enumerate(csv.DictReader(open(input_file)), start=1):
        if i == 1:
            for item in row:
                if 'I' in item:
                    num_count += 1
                if 'C' in item:
                    cat_count += 1
        break

    print("Numerical feature: {0}; Categorical feature: {1}\n".format(num_count, cat_count))

    counts = collections.defaultdict(lambda: [0, 0, 0])

    for i, row in enumerate(csv.DictReader(open(input_file)), start=1):
        label = row['target']
        for j in range(1, cat_count + 1):
            field = 'C{0}'.format(j)
            value = row[field]
            if label == '0':
                counts[field + ',' + value][0] += 1
            else:
                counts[field + ',' + value][1] += 1
            counts[field + ',' + value][2] += 1
        if i % 1000000 == 0:
            print('{0}m'.format(int(i / 1000000)))

    with open(output_file, 'w') as fout:
        fout.write('Field,Value,Neg,Pos,Total,Deviation\n')
        content = []
        for key, (neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
            if total < 10:
                continue
            deviation = round((float(pos / total) - 0.5) * 100, 2)
            content.append(key + ',' + str(neg) + ',' + str(pos) + ',' + str(total) + ',' + str(deviation) + '\n')
        content.reverse()
        for eachline in content:
            fout.write(eachline)


def extra_feature_pair(input_file, output_file, threshold1=200000, threshold2=1):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for index, eachline in enumerate(fin, start=1):
            if index > 2:
                line = eachline.strip().split(',')
                total = line[4]
                deviation = line[5]
                if int(total) >= threshold1:
                    fout.write(eachline)


def extract_topKfeature(input_file, topK, threshold=40):
    topK_feature = []
    with open(input_file, 'r') as fin:
        for index, eachline in enumerate(fin, start=1):
            line = eachline.strip().split(',')
            field = line[0]
            attribute = line[1]
            total = line[4]
            if total.isdigit():
                total = int(int(line[4])/10000)
                if topK:
                    if index > topK:
                        break
                    out_str = field + '-' + attribute
                    topK_feature.append(out_str)
                else:
                    if int(total) > threshold:
                        out_str = field + '-' + attribute
                        topK_feature.append(out_str)
    print(topK_feature)
    print('The topK features number is: {0}'.format(len(topK_feature)))
    return topK_feature


def read_freqent_feats(threshold=100):
    frequent_feats = set()
    for row in csv.DictReader(open(fc_trva_file)):
        if int(row['Total']) < threshold:
            continue
        frequent_feats.add(row['Field'] + '-' + row['Value'])
    return frequent_feats


def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (nr_bins - 1) + 1


if __name__ == '__main__':
    count(train_file, fc_trva_file)
    extra_feature_pair(fc_trva_file, fc_file, threshold1=1000000)
