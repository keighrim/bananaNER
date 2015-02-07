# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
extract useful suffixes from a dictionary of names

CS137B, programming assignment #1, Spring 2015
"""
import sys
import operator

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'krim'
__date__ = '2/6/2015'
__email__ = 'krim@brandeis.edu'

def get_suffix(dict_type):
    suffix = {}
    dict_file = dict_type + ".dict"
    with open(dict_file) as data:
        for line in data:
            line = line.strip().split()
            if len(line) > 1:
                try: 
                    suffix[line[-1]] += 1 
                except KeyError:
                    suffix[line[-1]] = 1
    top_suffixes = sorted(suffix.items(), key=operator.itemgetter(1), reverse=True)
    with open(dict_type + ".suff", "w") as outf:
        for suff in top_suffixes:
            outf.write("%s\t%s\n" % suff)

if __name__ == '__main__':
    
    get_suffix("org")

