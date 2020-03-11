#!/usr/bin/env python3
import sys
import csv
from os import path


usage = "USAGE: ./csv_convert.py <FILE PATH>"
if (len(sys.argv) != 2) or (not path.isfile(sys.argv[1])):
    print(usage)
    exit(1)

filename = sys.argv[1]
out_filename = filename.split('.')[:-1]
out_filename = '.'.join(out_filename) + '.csv'
with open(filename, 'r') as in_file:
    strip = (line.strip() for line in in_file)
    lines = (line.split()[:3] for line in strip if line[0].isdigit())
    with open(out_filename, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('epoch', 'depth', 'error'))
        writer.writerows(lines)
