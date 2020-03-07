import sys
import csv

filename = sys.argv[1]
out_filename = filename.split('.')[:-1]
out_filename = '.'.join(out_filename) + '.csv'
# TODO: some simple filename error checking
with open(filename, 'r') as in_file:
    strip = (line.strip() for line in in_file)
    lines = (line.split() for line in strip)
    with open(out_filename, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('epoch', 'depth', 'error'))
        writer.writerows(lines)
