# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
WRITE SOMETHING

CS137B, programming assignment #1, Spring 2015
"""
import sys

__author__ = ""
__email__ = ""

import collections
import os

CURPATH = os.getcwd()
PROJECT_PATH = os.path.sep.join(CURPATH.split(os.path.sep)[:-1])


class TaggerFrame():
    """TaggerFrame is a framework for tagging tokens from data file"""

    def __init__(self):
        self.sentences = None

        # all feature_functions should
        # 1. take no parameters
        # (use self.sentences and self.tokens())
        # 2. return a list or an iterable which has len of # number of tokens
        self.feature_functions = [self.postags, 
                                  self.first_word, 
                                  self.brown_100,
                                  self.is_banana,
                                  self.greater_ave_length]

    def read(self, input_filename):
        """load sentences in data file"""
        sentences = []
        sentence = []
        with open(input_filename) as in_file:
            for line in in_file:
                if line == "\n":
                    sentences.append(sentence)
                    sentence = []
                else:
                    try: 
                        sentence.append((line.split("\t")[1].strip(),
                                         line.split("\t")[2].strip(),
                                         line.split("\t")[3].strip()))
                    except:
                        sentence.append((line.split("\t")[1].strip(),
                                         line.split("\t")[2].strip(), ""))
        self.sentences = sentences

    def tokens(self):
        """Return a list of all tokens"""
        token_list = []
        for sent in self.sentences:
            for w_index, (word, _, _) in enumerate(sent):
                token_list.append((str(w_index), word))
        return token_list
        # return zip(range(len(self.sentences)),
        #            [w for sentence in self.sentences for (w, _, _) in sentence])

    def postags(self):
        """Return a list of all pos tags (in order)"""
        return [p for sentence in self.sentences for (_, p, _) in sentence]

    def biotags(self):
        """Return a list of all pos tags (in order)"""
        return [b for sentence in self.sentences for (_, _, b) in sentence]

    def feature_matrix(self, filename, train=True):
        """use this method to get all feature values and printed out as a file"""
        outf = open(filename, "w")

        features = self.get_features(train)
        for tok_index in range(len(features)):
            outf.write("\t".join(features[tok_index]) + "\n")
            try:
                if features[tok_index+1][0] == "0":
                    outf.write("\n")
            except KeyError:
                pass
        outf.close()

    def get_features(self, train=True):
        """traverse function list and get all values in a dictionary"""
        features = {}
        # unigram is the only default feature
        for i, (w_index, word) in enumerate(self.tokens()):
            features[i] = [w_index, word]

        # add gold bio tags in training
        if train:
            self.feature_functions.append(self.biotags)
        # traverse functions
        # note that all function should take no parameter and return an iterable 
        # which has length of the number of total tokens
        for fn in self.feature_functions:
            for num, feature in enumerate(fn()):
                features[num].append(feature)
                
        # remove gold tags when it's done
        if train: 
            self.feature_functions.remove(self.biotags)
        return features

    ######################################
    # currently not used
    @staticmethod
    def is_equal(token_list, sentence):
        compare = lambda a, b: collections.Counter(a) == collections.Counter(b)
        return compare(token_list, sentence)

    def get_sentence_tags(self, token_list):
        """using token_list to get the tags """
        for key in self.database.keys():
            sentence = self.database[key][0]
            if self.is_equal(token_list, sentence):
                return self.database[key]
        return None

    def get_token_tags(self, token_list):
        """using token_list to find the token-tag pair"""
        sentence, tags = self.get_sentence_tags(token_list)
        l = []
        for index in range(len(sentence)):
            l.append((sentence[index], tags[index]))
        return l

    def output_database(self):
        """traverse and print all the items in the database"""
        for key in self.database.keys():
            print key, self.database[key]

    def add_function(self, function_name):
        """build up a function list"""
        self.feature_functions.append(function_name)

    def extract(self):
        """traverse and execute the list of functions"""
        for function in self.feature_functions:
            function()

    # currently not used
    ########################################

    def first_word(self):
        """Checks for each word if it is the first word in a sentence"""
        word_list = []
        t = "first_word"
        f = "-first_word"
        for sent in self.sentences:
            if len(sent) > 0:
                word_list.append(t)
                word_list.extend([f] * (len(sent) - 1))
        return word_list

    def brown_50(self):
        return self.brown_cluster(50)

    def brown_100(self):
        return self.brown_cluster(100)

    def brown_150(self):
        return self.brown_cluster(150)

    def brown_200(self):
        return self.brown_cluster(200)

    def brown_250(self):
        return self.brown_cluster(250)

    def brown_300(self):
        return self.brown_cluster(300)

    def brown_400(self):
        return self.brown_cluster(400)

    def brown_500(self):
        return self.brown_cluster(500)

    def brown_600(self):
        return self.brown_cluster(600)

    def brown_700(self):
        return self.brown_cluster(700)

    def brown_800(self):
        return self.brown_cluster(800)

    def brown_900(self):
        return self.brown_cluster(900)

    def brown_1000(self):
        return self.brown_cluster(1000)

    def brown_cluster(self, num):
        """Gives words a feature based on their clusters. Can specify how
        many clusters to use: 50-300 by 50, 300-1000 by 100."""
        cluster_path \
            = os.path.join(PROJECT_PATH, 'dataset', 'clusters', 'paths_' + str(num))
        cluster_dict = {}
        with open(cluster_path) as cluster_file:
            for line in cluster_file:
                cluster, word, _ = line.split('\t')
                cluster_dict[word] = cluster

        word_list = []
        s = str(num) + "brown="
        for sent in self.sentences:
            for w, _, _ in sent:
                try:
                    word_list.append(s + cluster_dict[w])
                except KeyError:
                    word_list.append(s + "NONE")
        return word_list

    def greater_ave_length(self):
        """Calculates the average length of words in the corpus, then for each
        word checks if it is longer than the average length or not."""
        word_list = []
        total = 0
        num_words = 0
        for sent in self.sentences:
            num_words += len(sent)
            for w, _, _ in sent:
                total += len(w)
        average = total / num_words
        for sent in self.sentences:
            for w, _, _ in sent:
                if len(w) > average:
                    word_list.append(">_ave_length")
                else:
                    word_list.append("<=_ave_length")
        return word_list

    def is_banana(self):
        """Checks to see if a word is 'banana' or not."""
        word_list = []
        for sent in self.sentences:
            for w, _, _ in sent:
                if w.lower() == 'banana':
                    word_list.append("banana")
                else:
                    word_list.append("not_banana")
        return word_list

class NamedEntityReconizer(object):
    """
    NER class is a classifier to detect named entities 
    using TaggerFrame as feature extractor 
    and CRF++ as classification algorithm
    """
    
    def __init__(self):
        super(NamedEntityReconizer, self).__init__()
        os.chdir(PROJECT_PATH)
        self.fe = TaggerFrame()
        self.trainfile \
            = os.path.join('result', 'trainFeatureVector.txt')
        self.targetfile\
            = os.path.join('result', 'targetFeatureVector.txt')
        self.windows = sys.platform.startswith("win")
        if self.windows:
            self.crfppl \
                = os.path.join("crfpp_win", 'crf_learn')
            self.crfppc \
                = os.path.join("crfpp_win", 'crf_test')
        else:
            # TODO modify path and exec file name
            self.crfppl \
                = os.path.join("..", 'crfpp_win')
            self.crfppc \
                = os.path.join("..", 'crfpp_win', 'crf_test')
    
    def train(self, train_fn):
        self.fe.read(train_fn)
        self.fe.feature_matrix(self.trainfile)
        modelfile = os.path.join("result", "model")
        templatefile = os.path.join("result", "template")
        if self.windows:
            command = r'%s "%s" "%s" "%s"' %\
                      (self.crfppl, templatefile, self.trainfile, modelfile)
        else: 
            # TODO something for linux/mac
            command = ""
            pass
        os.system(command)

    def classify(self, target_fn):
        self.fe.read(target_fn)
        self.fe.feature_matrix(self.targetfile, False)
        modelfile = os.path.join("result", "model")
        resultfile = os.path.join("result", "result.txt")
        if self.windows:
            os.system('del "' + resultfile + '"')
            command = r'%s -m "%s" "%s" >> "%s"' % \
                      (self.crfppc, modelfile, self.targetfile, resultfile)
        else:
            os.system('rm "' + resultfile + '"')
            # TODO something for linux/mac
            command = ""
            pass
        os.system(command)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="name of train set file",
        default=os.path.join(PROJECT_PATH, 'dataset', 'train.gold')
    )
    # parser.add_argument(
    #     "-o",
    #     help="name of a file to print out extracted features",
    #     default=os.path.join(PROJECT_PATH, 'result', 'output.txt')
    # )
    parser.add_argument(
        "-t",
        help="name of target file, if not given, program will ask users after training",
        default=None
    )
    args = parser.parse_args()
    ner = NamedEntityReconizer()
    ner.train(args.i)
    if args.t is None:
        # TODO write input() to get target file name
        target = input("enter file name with its full path: ")
        pass
    else:
        target = args.t
    ner.classify(target)

