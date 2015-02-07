# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
WRITE SOMETHING

CS137B, programming assignment #1, Spring 2015
"""
import re
import sys

__author__ = ["Keigh Rim", "Todd Curcuru", "Yalin Liu"]
__date__ = "2/1/2015"
__email__ = ['krim@brandeis.edu', 'tcurcuru@brandeis.edu', 'yalin@brandeis.edu']

import os

CURPATH = os.getcwd()
PROJECT_PATH = os.path.sep.join(CURPATH.split(os.path.sep)[:-1])
RES_PATH = os.path.join(PROJECT_PATH, "resources")
DICT_PATH = os.path.join(RES_PATH, "dicts")
FREQ_PATH = os.path.join(RES_PATH, "freqCounts")
CLUSTER_PATH = os.path.join(RES_PATH, "clusters")


class FeatureTagger():
    """FeatureTagger is a framework for tagging tokens from data file"""

    def __init__(self):
        # sentences is a list of triples [(word, postag, biotag)]
        # if bio is not given (raw file) third elements are empty strings
        self.sentences = None

        # df and tf is frequency counts, document freq, term freq, respectively
        self.df = {}
        self.tf = {}
        self.cdf = {}
        self.ctf = {}

        # dicts will store name dictionaries
        self.dicts = {}
        self.org_suffixes = []

        # all feature_functions should
        # 1. take no parameters
        # (use self.sentences and self.tokens())
        # 2. return a list or an iterable which has len of # number of tokens
        self.feature_functions = [self.postags, #2
                                  self.zone,  #3
                                  self.bias,  #4
                                  self.first_word,  #5
                                  self.initcap, #6
                                  self.one_cap, #7
                                  self.allcap,  #8
                                  self.contain_digit, #9
                                  self.two_digit, #10
                                  self.four_digit,  #11
                                  self.digit_period,  #12
                                  self.digit_slash, #13
                                  self.dollar,  #14
                                  self.percent, #15
                                  self.greater_ave_length,  #16
                                  self.initcap_period,  #17
                                  self.allcap_period, #18
                                  self.brown_50,  #19
                                  self.brown_100, #20
                                  self.brown_150, #21
                                  self.brown_200, #22
                                  self.brown_250, #23
                                  self.brown_300, #24
                                  self.brown_400, #25
                                  self.brown_500, #26
                                  self.brown_600, #27
                                  self.brown_700, #28
                                  self.brown_800, #29
                                  self.brown_900, #30
                                  self.brown_1000,  #31
                                  self.hyphen,  #32
                                  self.term_freq, #33
                                  self.docu_freq, #34
                                  self.ctf_50,  #35
                                  self.ctf_100, #36
                                  self.ctf_150, #37
                                  self.ctf_200, #38
                                  self.ctf_250, #39
                                  self.ctf_300, #40
                                  self.ctf_400, #41
                                  self.ctf_500, #42
                                  self.ctf_600, #43
                                  self.ctf_700, #44
                                  self.ctf_800, #45
                                  self.ctf_900, #46
                                  self.ctf_1000,  #47
                                  self.cdf_50,  #48
                                  self.cdf_100, #49
                                  self.cdf_150, #50
                                  self.cdf_200, #51
                                  self.cdf_250, #52
                                  self.cdf_300, #53
                                  self.cdf_400, #54
                                  self.cdf_500, #55
                                  self.cdf_600, #56
                                  self.cdf_700, #57
                                  self.cdf_800, #58
                                  self.cdf_900, #59
                                  self.cdf_1000,  #60
                                  self.seq_caps, #61
                                  self.dict_geo,    #62
                                  self.dict_person, #63
                                  self.dict_org,    #64
                                  self.dict_other,    #65
                                  self.prox_org_suff #66
        ]

    def read(self, input_filename):
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
        self.sentences = sentences
        self.populate_freq(300)
        self.populate_dict()

    def populate_freq(self, cluster_size):
        """
        Load up frequency counts from external files
        df, tf are document frequency and term frequency of each term
        cdf, ctf are df and tf of each cluster ID of each term
        :param cluster_size: size of cluster to use
        """
        with open(os.path.join(FREQ_PATH, "c%i_document.freq" % cluster_size)) as cdf, \
                open(os.path.join(FREQ_PATH, "c%i_term.freq" % cluster_size)) as ctf, \
                open(os.path.join(FREQ_PATH, "document.freq")) as df, \
                open(os.path.join(FREQ_PATH, "term.freq")) as tf:
            self.cdf = self.read_freq(cdf)
            self.ctf = self.read_freq(ctf)
            self.df = self.read_freq(df)
            self.tf = self.read_freq(tf)

    @staticmethod
    def read_freq(freq_file):
        """reads frequency count file and make a dict from it"""
        freq = {}
        for line in freq_file:
            freq[line.split("\t")[0]] = int(line.split("\t")[1].strip())
        return freq

    @staticmethod
    def freq_rank(freq_dict):
        """
        grade frequency counts with three ranks
        top 25% get 'High'
        bottom 50% get 'Low'
        rest get 'Mid'
        :param freq_dict: 
        :return: dict with freq counts as keys, their rank as values
        """
        sorted_freq \
            = sorted(set([f for _, f in freq_dict.iteritems()]), reverse=True)
        how_freq = {}
        for freq in sorted_freq[:len(sorted_freq)/4]:
            how_freq[freq] = "High"
        for freq in sorted_freq[len(sorted_freq)/4:len(sorted_freq)/2]:
            how_freq[freq] = "Mid"
        for freq in sorted_freq[len(sorted_freq)/2:]:
            how_freq[freq] = "Low"
        return how_freq

    def docu_freq(self):
        """feature function: return the rank of df of each token"""
        tag = []
        s = "DocuFreq="
        f_rank = self.freq_rank(self.df)
        for sent in self.sentences:
            for w, _, _ in sent:
                try:
                    tag.append(s + f_rank[self.df[w.lower()]])
                except KeyError:
                    tag.append(s + "OOV")
        return tag

    def term_freq(self):
        """feature function: return the rank of tf of each token"""
        tag = []
        s = "TermFreq="
        f_rank = self.freq_rank(self.tf)
        for sent in self.sentences:
            for w, _, _ in sent:
                try:
                    tag.append(s + f_rank[self.tf[w.lower()]])
                except KeyError:
                    tag.append(s + "OOV")
        return tag

    # feature functions: return the rank of cdf of each token of given cluster size
    def cdf_50(self): return self.cluster_docu_freq(50)
    def cdf_100(self): return self.cluster_docu_freq(100)
    def cdf_150(self): return self.cluster_docu_freq(150)
    def cdf_200(self): return self.cluster_docu_freq(200)
    def cdf_250(self): return self.cluster_docu_freq(250)
    def cdf_300(self): return self.cluster_docu_freq(300)
    def cdf_400(self): return self.cluster_docu_freq(400)
    def cdf_500(self): return self.cluster_docu_freq(500)
    def cdf_600(self): return self.cluster_docu_freq(600)
    def cdf_700(self): return self.cluster_docu_freq(700)
    def cdf_800(self): return self.cluster_docu_freq(800)
    def cdf_900(self): return self.cluster_docu_freq(900)
    def cdf_1000(self): return self.cluster_docu_freq(1000)

    def cluster_docu_freq(self, cluster_size):
        """given cluster granularity,
         replace words into cluster representation and return its document frequency"""
        self.populate_freq(cluster_size)
        cluster_path = os.path.join(
            CLUSTER_PATH, 'paths_' + str(cluster_size))
        cluster_dict = {}
        with open(cluster_path) as cluster_file:
            for line in cluster_file:
                cluster, word, _ = line.split('\t')
                cluster_dict[word] = cluster
        tag = []
        s = "C%iDocuFreq=" % cluster_size
        f_rank = self.freq_rank(self.cdf)
        for sent in self.sentences:
            for w, _, _ in sent:
                try:
                    tag.append(s + f_rank[self.cdf[cluster_dict[w]]])
                except KeyError:
                    tag.append(s + "OOV")
        return tag

    # feature functions: return the rank of ctf of each token of given cluster size
    def ctf_50(self): return self.cluster_term_freq(50)
    def ctf_100(self): return self.cluster_term_freq(100)
    def ctf_150(self): return self.cluster_term_freq(150)
    def ctf_200(self): return self.cluster_term_freq(200)
    def ctf_250(self): return self.cluster_term_freq(250)
    def ctf_300(self): return self.cluster_term_freq(300)
    def ctf_400(self): return self.cluster_term_freq(400)
    def ctf_500(self): return self.cluster_term_freq(500)
    def ctf_600(self): return self.cluster_term_freq(600)
    def ctf_700(self): return self.cluster_term_freq(700)
    def ctf_800(self): return self.cluster_term_freq(800)
    def ctf_900(self): return self.cluster_term_freq(900)
    def ctf_1000(self): return self.cluster_term_freq(900)

    def cluster_term_freq(self, cluster_size):
        """given cluster granularity,
         replace words into cluster representation and return its term frequency"""
        self.populate_freq(cluster_size)
        cluster_path = os.path.join(
            CLUSTER_PATH, 'paths_' + str(cluster_size))
        cluster_dict = {}
        with open(cluster_path) as cluster_file:
            for line in cluster_file:
                cluster, word, _ = line.split('\t')
                cluster_dict[word] = cluster

        tag = []
        s = "C%iTermFreq=" % cluster_size
        f_rank = self.freq_rank(self.ctf)
        for sent in self.sentences:
            for w, _, _ in sent:
                try:
                    tag.append(s + f_rank[self.ctf[cluster_dict[w]]])
                except KeyError:
                    tag.append(s + "OOV")
        return tag

    def tokens(self):
        """Return a list of all tokens with their index in the sent they're belong"""
        token_list = []
        for sent in self.sentences:
            for w_index, (word, _, _) in enumerate(sent):
                token_list.append((str(w_index), word))
        return token_list

    def postags(self):
        """feature function: Return a list of all pos tags (in order)"""
        return [p for sentence in self.sentences for (_, p, _) in sentence]

    def biotags(self):
        """Return a list of all pos tags (in order)"""
        return [b for sentence in self.sentences for (_, _, b) in sentence]

    def feature_matrix(self, filename, train=True):
        """use this method to get all feature values and printed out as a file"""
        with open(filename, "w") as outf, \
                open(os.path.join("result", "template"), "w") as template:

            features = self.get_features(train)
            for tok_index in range(len(features)):
                outf.write("\t".join(features[tok_index]) + "\n")
                try:
                    if features[tok_index+1][0] == "0":
                        outf.write("\n")
                except KeyError:
                    pass
            with open("scripts/template_addendum.txt", "r") as temp:
                for line in temp:
                    template.write(line)

    def get_features(self, train=True):
        """traverse function list and get all values in a dictionary"""
        features = {}
        # unigram is the only default feature
        for i, (w_index, word) in enumerate(self.tokens()):
            features[i] = [w_index, word]

        # add gold bio tags while training
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

    def bias(self):
        """feature function: dummy bias feature that gives 1 to all tokens"""
        return ["bias"] * len(self.tokens())

    def first_word(self):
        """feature functin: 
        Checks for each word if it is the first word in a sentence"""
        word_list = []
        t = "first_word"
        f = "-first_word"
        for sent in self.sentences:
            if len(sent) > 0:
                word_list.append(t)
                word_list.extend([f] * (len(sent) - 1))
        return word_list

    # feature functions: return cluster ID of each token, given cluster size
    def brown_50(self): return self.brown_cluster(50)
    def brown_100(self): return self.brown_cluster(100)
    def brown_150(self): return self.brown_cluster(150)
    def brown_200(self): return self.brown_cluster(200)
    def brown_250(self): return self.brown_cluster(250)
    def brown_300(self): return self.brown_cluster(300)
    def brown_400(self): return self.brown_cluster(400)
    def brown_500(self): return self.brown_cluster(500)
    def brown_600(self): return self.brown_cluster(600)
    def brown_700(self): return self.brown_cluster(700)
    def brown_800(self): return self.brown_cluster(800)
    def brown_900(self): return self.brown_cluster(900)
    def brown_1000(self): return self.brown_cluster(1000)

    def brown_cluster(self, num):
        """Gives words a feature based on their clusters. Can specify how
        many clusters to use: 50-300 by 50, 300-1000 by 100."""
        cluster_file = os.path.join(CLUSTER_PATH, 'paths_' + str(num))
        cluster_dict = {}
        with open(cluster_file) as cluster_file:
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

    def zone(self):
        """feature function: return a list of each token's zone information"""
        tag = []
        ZONE = "zone="
        txt_zone = ZONE + "TXT"
        title_zone = ZONE + "HL"
        byline_zone = ZONE + "DATELINE"
        dd_zone = ZONE + "DD"
        zone = -1
        for sent in self.sentences:
            nyt = re.search(r"NYT[0-9]{8}", sent[0][0])
            if nyt:
                zone = 3
            if re.search(r"APW[0-9]{8}", sent[0][0]):
                zone = 2
            for w, _, _ in sent:
                if zone == 2:
                    tag.append(byline_zone)
                elif zone == 1:
                    tag.append(dd_zone)
                elif zone == 0:
                    tag.append(title_zone)
                else:
                    tag.append(txt_zone)
            if nyt:
                zone -= 2
            else:
                zone -= 1
        return tag


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

    def initcap(self):
        """feature function: check if a token starts with capital letter"""
        tag = []
        t = "Initcap"
        f = "-Initcap"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w[0].isupper():
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def initcap_period(self):
        """feature function: 
        check if a token starts with capital letter and ends with a period"""
        tag = []
        t = "InitcapPeriod"
        f = "-InitcapPeriod"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w[0].isupper() and w[-1] == '.':
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def one_cap(self):
        """feature function: check if a token consists of only 1 capital letter"""
        tag = []
        t = "OneCap"
        f = "-OneCap"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w.isupper() and len(w) == 1:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def allcap(self):
        """feature function: see if a token consists of all capital letters"""
        tag = []
        t = "Allcap"
        f = "-Allcap"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w.isupper():
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def allcap_period(self):
        """feature function:
        see if a token consists of all capital letters and ends with a period"""
        tag = []
        t = "AllcapPeriod"
        f = "-AllcapPeriod"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w.isupper() and w[-1] == ".":
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def contain_digit(self):
        """feature function: check if a token contains one or more digits"""
        tag = []
        t = "wDigit"
        f = "woDigit"
        for sent in self.sentences:
            for w, _, _ in sent:
                if len(filter(lambda x: x.isdigit(), w[:])) > 1:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def two_digit(self):
        """feature funtion: check if a token consists of only two digits"""
        tag = []
        t = "TwoDigit"
        f = "-TwoDigit"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w.isdigit() and len(w) == 2:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def four_digit(self):
        """feature funtion: check if a token consists of only four digits"""
        tag = []
        t = "FourDigit"
        f = "-FourDigit"
        for sent in self.sentences:
            for w, _, _ in sent:
                if w.isdigit() and len(w) == 4:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def digit_slash(self):
        """feature function: check if a token consists of only digits and slahes"""
        tag = []
        t = "DigitSlash"
        f = "-DigitSlash"
        isslash = lambda x: x == "/"
        is_slash_or_digit = lambda x: x.isdigit() or isslash(x)
        for sent in self.sentences:
            for w, _, _ in sent:
                if len(filter(is_slash_or_digit, [c for c in w])) == len(w):
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def dollar(self):
        """feature function: check a dollar symbol is in a token"""
        tag = []
        t = "Dollar"
        f = "-Dollar"
        for sent in self.sentences:
            for w, _, _ in sent:
                if "$" in w:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def percent(self):
        """feature function: check a percent symbol is in a token"""
        tag = []
        t = "Percent"
        f = "-Percent"
        for sent in self.sentences:
            for w, _, _ in sent:
                if "%" in w:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def hyphen(self):
        """feature function: check is a token is hyphenated one"""
        tag = []
        t = "Hyphen"
        f = "-Hyphen"
        for sent in self.sentences:
            for w, _, _ in sent:
                if "%" in w:
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def digit_period(self):
        """feature function: check if a token consists of only digits and periods"""
        tag = []
        t = "DigitPeriod"
        f = "-DigitPeriod"
        isperiod = lambda x: x == "."
        is_slash_or_digit = lambda x: x.isdigit() or isperiod(x)
        for sent in self.sentences:
            for w, _, _ in sent:
                if len(filter(is_slash_or_digit, [c for c in w])) == len(w):
                    tag.append(t)
                else:
                    tag.append(f)
        return tag

    def seq_caps(self):
        """feature function: 
        return true only when previous, current and next tokens are all init_cap"""
        tag = []
        t = "SeqInit"
        f = "-SeqInit"
        initcap = lambda x: len(x) > 2 and x[0].isupper()
        for sent in self.sentences:
            words = [w for w, _, _ in sent]
            for i in range(len(words)):
                if len(words) < 2:
                    tag.append(f)
                elif i == 0 and len(filter(initcap, (words[i], words[i+1]))) == 2:
                    tag.append(t)
                elif i == len(words) - 1 and len(filter(initcap, (words[i-1], words[i]))) == 2:
                    tag.append(t)
                else:
                    try:
                        if len(filter(initcap, (words[i-1], words[i], words[i+1]))) == 3:
                            tag.append(t)
                        else:
                            tag.append(f)
                    except IndexError:
                        tag.append(f)
                    initcap(words[i-1]) and initcap([words[i]])
        return tag

    def populate_dict(self):
        """Populate dictionaries using external files"""
        for filename in filter(lambda x: x.endswith(".dict"), os.listdir(DICT_PATH)):
            dict_type = filename.split(".")[0]
            self.dicts[dict_type] = []
            with open(os.path.join(DICT_PATH, filename)) as d:
                for line in d:
                    if line != "\n":
                        self.dicts[dict_type].append(line.strip().lower())
            # now load up useful suffix dict
            if dict_type == "org":  # only organization names have useful suffixes
                with open(os.path.join(DICT_PATH, dict_type + ".suff")) as suff:
                    for line in suff:
                        if line != "\n":
                            self.org_suffixes.append(line.strip().lower())

    # feature functions: check if words are in each dictionary
    def dict_person(self): return self.in_dict("person")
    def dict_geo(self): return self.in_dict("geo")
    def dict_org(self): return self.in_dict("org")
    def dict_other(self): return self.in_dict("other")

    def in_dict(self, typ):
        """See each token is in a certain dictionary"""
        tag = []
        s = "In_" + typ
        d = self.dicts[typ]
        for num, sent in enumerate(self.sentences):
            tokens = [w for w, _, _ in sent]
            i = 0
            if typ == "person":
                for token in tokens:
                    if token.lower() in d:
                        tag.append(s)
                    else:
                        tag.append("-" + s)
                pass
            else:
                while i < len(tokens):
                    for j in range(i + 1, len(tokens)):
                        if " ".join(map(lambda x: x.lower(), tokens[i:j])) in d:
                            tag.extend([s] * (j - i))
                            i = j
                            break
                    tag.append("-" + s)
                    i += 1
        return tag

    def prox_org_suff(self):
        """feature function: see if there's a frequent suffix for org names
        in coming 5 tokens of each token"""
        tag = []
        t = "ComingOrgSuff"
        f = "-ComingOrgSuff"
        for sent in self.sentences:
            words = [w for w, _, _ in sent]
            s_tag = [f] * len(words)
            for i in range(len(words) - 2):
                for word in words[i+1:i+5]:
                    if word in self.org_suffixes:
                        s_tag[i] = t
                        break
            tag.extend(s_tag)
        return tag


class NamedEntityRecognizer(object):
    """
    NER class is a classifier to detect named entities 
    using TaggerFrame as feature extractor 
    and CRF++ as classification algorithm
    """

    def __init__(self):
        super(NamedEntityRecognizer, self).__init__()
        os.chdir(PROJECT_PATH)
        self.ft = FeatureTagger()
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
            # TODO modify paths and exec file names for mac/linux
            self.crfppl \
                = os.path.join("..", 'crfpp_win')
            self.crfppc \
                = os.path.join("..", 'crfpp_win', 'crf_test')

    def train(self, train_fn):
        """train crf++ module with a given data file"""
        self.ft.read(train_fn)
        self.ft.feature_matrix(self.trainfile)
        modelfile = os.path.join("result", "model")
        templatefile = os.path.join("result", "template")
        if self.windows:
            command = r'%s "%s" "%s" "%s"' %\
                      (self.crfppl, templatefile, self.trainfile, modelfile)
        else:
            # TODO something for linux/mac
            command = ""
            pass
        # TODO change this to use subprocess module
        os.system(command)

    def classify(self, target_fn):
        """Run crfpp classifier to classify target file"""
        self.ft.read(target_fn)
        self.ft.feature_matrix(self.targetfile, False)
        modelfile = os.path.join("result", "model")
        if not os.path.isfile(modelfile):
            raise Exception("Model not found.")
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
        # TODO change this to use subprocess module
        os.system(command)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="name of train set file",
        default=os.path.join(PROJECT_PATH, 'dataset', 'train.gold')
    )
    parser.add_argument(
        "-t",
        help=
        "name of target file, if not given, program will ask users after training",
        default=None
    )
    args = parser.parse_args()

    ner = NamedEntityRecognizer()
    ner.train(args.i)
    if args.t is None:
        try:
            target = input(
                "enter a test file name with its path\n" 
                + "(relative or full, default: dataset/dev.raw): ")
        # if imput is empty
        except SyntaxError:
            target = "dataset/dev.raw"
    else:
        target = args.t
    ner.classify(target)

    # run eval code
    testfile = target.split("/")[-1].split(".")[0]
    # TODO change to subprocess module
    os.system(
        'python scripts/evaluate-head.py "dataset/%s.gold" "result/result.txt"'
        % testfile)
