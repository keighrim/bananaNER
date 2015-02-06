# /usr/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import operator

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'krim'
__date__ = '2/5/2015'
__email__ = 'krim@brandeis.edu'

"""
This program is to:
count frequencies of terms and their brown numbers
"""

def count_frequencies(input_filename):
    """Count document frequency of each word's cluster number(words not lowered)"""
    documents = []
    sentences = []
    tokens = []

    # first read up data file (again but ina different format)
    with open(input_filename) as data:
        for line in data:
            if line == "\n":
                if len(tokens) > 0:
                    sentences.append(tokens)
                    tokens = []
            else:
                w = line.split("\t")[1].strip()
                p = line.split("\t")[2].strip()
                b = line.split("\t")[3].strip()
                if re.search(r"[A-Z]{3}[0-9]{8}", w) and len(sentences) > 0:
                    documents.append(sentences)
                    sentences = []
                tokens.append((w, p, b))
        sentences.append(tokens)
        documents.append(sentences)

        # count document frequency
        df_count = {}
        for document in documents:
            for word in set([w.lower()
                             for sentence in document
                             for w, _, _ in sentence]):
                try:
                    df_count[word] += 1
                except KeyError:
                    df_count[word] = 1
        df = sorted(df_count.items(),
                         key=operator.itemgetter(1), reverse=True)

        # count term frequency
        tf_count = {}
        for term in [w.lower()
                     for doc in documents for sent in doc for w, _, _ in sent]:
            try:
                tf_count[term] += 1
            except KeyError:
                tf_count[term] = 1
        tf = sorted(tf_count.items(),
                         key=operator.itemgetter(1), reverse=True)

        with open("document.freq", "w") as dff, open('term.freq', "w") as tff:
            for freq in tf:
                tff.write("%s\t%s\n" % freq)
            for freq in df:
                dff.write("%s\t%s\n" % freq)
                
def count_cluster_frequencies(input_filename, cluster_size):
    """Count document frequency of each word (lowered)"""
    documents = []
    sentences = []
    tokens = []

    # first read up data file
    with open(input_filename) as data:
        for line in data:
            if line == "\n":
                if len(tokens) > 0:
                    sentences.append(tokens)
                    tokens = []
            else:
                w = line.split("\t")[1].strip()
                p = line.split("\t")[2].strip()
                b = line.split("\t")[3].strip()
                if re.search(r"[A-Z]{3}[0-9]{8}", w) and len(sentences) > 0:
                    documents.append(sentences)
                    sentences = []
                tokens.append((w, p, b))
        sentences.append(tokens)
        documents.append(sentences)

        # then read up cluster file and count term freq (tf is in cluster file)
        cluster_file = "../dataset/clusters/paths_" + str(cluster_size)
        cluster_dict = {}
        cluster_tf_count = {}
        with open(cluster_file) as cluster:
            for line in cluster:
                line = line.split("\t")
                cluster_dict[line[1]] = line[0]
                try:
                    cluster_tf_count[line[0]] += int(line[2])
                except KeyError:
                    cluster_tf_count[line[0]] = int(line[2])
        cluster_tf = sorted(cluster_tf_count.items(),
                                 key=operator.itemgetter(1), reverse=True)

        # count document frequency
        cluster_df_count = {}
        for document in documents:
            for word in set([w
                             for sentence in document
                             for w, _, _ in sentence]):
                try:
                    cluster_df_count[cluster_dict[word]] += 1
                except KeyError:
                    cluster_df_count[cluster_dict[word]] = 1
        cluster_df = sorted(cluster_df_count.items(),
                                 key=operator.itemgetter(1), reverse=True)

        with open("c%i_document.freq" % cluster_size, "w") as cdf, \
                open("c%i_term.freq" % cluster_size, "w") as ctf:
            for freq in cluster_tf:
                ctf.write("%s\t%s\n" % freq)
            for freq in cluster_df:
                cdf.write("%s\t%s\n" % freq)

if __name__ == '__main__':
    count_frequencies("../dataset/all.gold")
    for i in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]:
        count_cluster_frequencies("../dataset/all.gold", i)