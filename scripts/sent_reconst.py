# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
reconstruct sentences from a given data file

CS137B, programming assignment #1, Spring 2015
"""
import re

__author__ = 'Keigh Rim'
__date__ = '2/1/2015'
__email__ = 'krim@brandeis.edu'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i",
        help="name of a data file"
    )
    parser.add_argument(
        "--o",
        help="name of the output file"
    )
    args = parser.parse_args()
    
    path = "../dataset/"
    sent = ""
    tags = ""
    with open(path+args.i) as in_file, open("../" + args.o, 'w') as out_file:
        for line in in_file:
            if re.search(r"^\s+$", line):
                sent += "\n"
                tags += "\n"
                out_file.write(sent)
                out_file.write(tags)
                out_file.write("\n")
                sent = ""
                tags = ""
            else:
                sent += line.split("\t")[1] + "\t"
                tags += line.split("\t")[2] + "\t"

