# -*- coding:utf-8 -*-

import argparse
import collections
import csv
import sys

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
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

sys.stderr.write("Numerical feature: {0}; Categorical feature: {1}\n".format(num_count, cat_count))

counts = collections.defaultdict(lambda: [0, 0, 0])

for i, row in enumerate(csv.DictReader(open(args['csv_path'])), start=1):
    label = row['target']
    for j in range(1, cat_count+1):
        field = 'C{0}'.format(j)
        value = row[field]
        if label == '0':
            counts[field+','+value][0] += 1
        else:
            counts[field+','+value][1] += 1
        counts[field+','+value][2] += 1
    if i % 1000000 == 0:
        sys.stderr.write('{0}m\n'.format(int(i/1000000)))

print('Field,Value,Neg,Pos,Total,Ratio')
for key, (neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
    if total < 10:
        continue
    ratio = abs(round(float(pos)/total, 5) - 0.5)
    print(key + ',' + str(neg) + ',' + str(pos) + ',' + str(total) + ',' + str(ratio))
