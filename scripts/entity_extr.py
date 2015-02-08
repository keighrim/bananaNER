# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to
extract named entities from an annotated data file

CS137B, programming assignment #1, Spring 2015
"""
import sys

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'krim'
__date__ = '2/6/2015'
__email__ = 'krim@brandeis.edu'


def read(input_filename):
    """load sentences from data file"""
    sentences = []
    sentence = []
    with open(input_filename) as in_file:
        for line in in_file:
            if re.search(r"^\s+$", line):
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
    
    # we'll use 4 dictionaries; ORG, GEO, PERSON, OTHER
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
            # for person names, do not concatenate
            if typ == "PER":
                if len(entity) > 0:
                    cur.append(entity)
                    entity = ""
                person.append(w)
            # else, keep track of "I" tagged words and concatenate
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
    # write out lists to coresponding files
    with open("org.extr", "w") as orgf, \
            open("other.extr", "w") as otherf, \
            open("person.extr", "w") as personf, \
            open("geo.extr", "w") as geof:
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

