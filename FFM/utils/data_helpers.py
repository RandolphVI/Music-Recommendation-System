# -*- coding:utf-8 -*-

import csv
import math
import hashlib
import pandas as pd


def view_file(file_name):
    with open(file_name, 'r') as f:
        for index, eachline in enumerate(f, start=1):
            print(eachline)
            if index > 5:
                break


def reverse(input_file, output_file):
    with open(input_file, 'rb') as fin:
        content = []
        for index, eachline in enumerate(fin, start=1):
            if index > 2:
                content.append(eachline)
        content.reverse()
        with open(output_file, 'wb') as fout:
            for item in content:
                fout.write(item)


def extra_feature_pair(input_file, output_file, threshold1=0.01, threshold2=200000):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for index, eachline in enumerate(fin, start=1):
            line = eachline.strip().split(',')
            total = line[4]
            ratio = line[5]
            if int(total) >= threshold2:
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
    for row in csv.DictReader(open('./data/fc.trva.top.txt')):
        if int(row['Total']) < threshold:
            continue
        frequent_feats.add(row['Field'] + '-' + row['Value'])
    return frequent_feats


def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (nr_bins - 1) + 1


if __name__ == '__main__':
    # view_file('reverse.txt')
    reverse('./data/fc.trva.top.txt', './data/fc.trva.txt')
    extra_feature_pair('./data/fc.trva.txt', './data/fc.txt', threshold2=1000000)
    pass
