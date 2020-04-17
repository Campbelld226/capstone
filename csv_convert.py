#!/usr/bin/env python3
import sys
import csv
from os import path


usage = "USAGE: ./csv_convert.py <FILE PATH> <OUT_PATH>"
if (len(sys.argv) != 3) or (not path.isfile(sys.argv[1])) or (not path.isdir(sys.argv[2])):
    print(usage)
    exit(1)

filename = sys.argv[1]
out_path = sys.argv[2]

out_filename = filename.split('.')[:-1]
out_filename = '.'.join(out_filename) + '.csv'
out_filename = out_path + '/' + out_filename.split('/')[-1]
with open(filename, 'r') as in_file:
    strip = (line.strip() for line in in_file if line[0].isdigit())
    lines = (line.split()[:3] for line in strip)
    with open(out_filename, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('epoch', 'depth', 'error'))
        writer.writerows(lines)
