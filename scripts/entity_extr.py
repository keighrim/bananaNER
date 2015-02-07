# /usr/bin/python
# -*- coding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'krim'
__date__ = '2/6/2015'
__email__ = 'krim@brandeis.edu'

"""
This program is to
"""

def read(input_filename):
    """load sentences from data file"""
    sentences = []
    sentence = []
    with open(input_filename) as in_file:
        for line in in_file:
            if line == "\n":
                if not prev_empty:
                    sentences.append(sentence)
                    sentence = []
                prev_empty = True
            else:
                try:
                    sentence.append((line.split("\t")[1].strip(),
                                     line.split("\t")[2].strip(),
                                     line.split("\t")[3].strip()))
                except IndexError:
                    sentence.append((line.split("\t")[1].strip(),
                                     line.split("\t")[2].strip(), ""))
                prev_empty = False
    return sentences

def find_entities(sents):
    org = []
    geo = []
    other = []
    person = []
    entity = ""
    for sent in sents:
        for w, _, b in sent:
            try:
                bio = b.split("-")[0]
                typ = b.split("-")[1]
            except IndexError:
                bio = "O"
                typ = ""
            if typ == "PER":
                if len(entity) > 0:
                    cur.append(entity)
                    entity = ""
                person.append(w)
            else:
                if bio == "B":
                    if len(entity) > 0:
                        cur.append(entity)
                    entity = w
                    if typ == "ORG":
                        cur = org
                    elif typ == "LOC" or typ == "GPE":
                        cur = geo
                    else:
                        cur = other
                elif bio == "I":
                    entity += " " + w
                else:
                    if len(entity) > 0:
                        cur.append(entity)
                        entity = ""
    with  open("org.extr", "w") as orgf, open("other.extr", "w") as otherf, open("person.extr", "w") as personf, open("geo.extr", "w") as geof:
        for o in org:
            orgf.write(o + "\n")
        for ot in other:
            otherf.write(ot + "\n")
        for p in person:
            personf.write(p + "\n")
        for g in geo:
           geof.write(g + "\n")
        
if __name__ == '__main__':
    # tempted to use all.gold...
    find_entities(read("../dataset/train.gold"))
            

