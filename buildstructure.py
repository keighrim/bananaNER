import collections
import os
path = os.getcwd()

class Tagger_frame():
    
    def __init__(self,file_name):
        #self.inpath = '../dataset/'+file_name
        #self.outpath = '../result/output.txt'
        self.inpath = os.path.join(path, 'dataset', file_name)
        self.outpath = os.path.join(path, 'result', 'output.txt')
        self.input = open(self.inpath)
        self.output = open(self.outpath,'w')
        self.database = {}
        self.feature_functions = []
        
    def read(self):
        '''load the data into database'''
        key = 0
        words = []
        tags = []
        
        for line in self.input:
            if line == '\n':
                value=(words,tags)
                self.database[key]= value
                words = []
                tags = []
                key += 1
                
            else:
                list = line.split()
                words.append(list[1])
                tags.append(list[2])
                
    def is_equal(self,token_list,sentence):
        compare = lambda x, y:collections.Counter(x) == collections.Counter(y)
        return compare(token_list,sentence)
                
    def get_sentence_tags(self, token_list):
        '''using token_list to get the tags '''
        for key in self.database.keys():
            sentence = self.database[key][0]
            if self.is_equal(token_list, sentence):
                return self.database[key]
        return None       
    
    def get_token_tags(self,token_list):
        '''using token_list to find the token-tag pair'''
        sentence,tags=self.get_sentence_tags(token_list)
        list = []
        for index in range(len(sentence)):
            list.append((sentence[index],tags[index]))
        return list
        
    def output_database(self):
        '''traverse and print all the items in the database'''
        for key in self.database.keys():
            print key, self.database[key]
         
    def add_function(self,function_name):
        '''build up a function list'''
        self.feature_functions.append(function_name)
        
    def extract(self):
        '''traverse and execute the list of functions'''
        for function in self.feature_functions:
            function()
            
    def first_word(self):
        """Checks for each word if it is the first word in a sentence"""
        word_list = []
        self.input.seek(0)
        for line in self.input:
            if line != '\n':
                if line.split('\t')[0] == "0":
                    word_list.append(True)
                else:
                    word_list.append(False)
        return word_list
        
    def brown_cluster(self, num):
        """Gives words a feature based on their clusters. Can specify how
        many clusters to use: 50-300 by 50, 300-1000 by 100."""
        cluster_path = os.path.join(path, 'dataset', 'clusters', 'paths_' + str(num))
        cluster_dict = {}
        with open(cluster_path) as cluster_file:
            for line in cluster_file:
                split = line.split('\t')
                cluster_dict[split[1]] = split[0]
                
        word_list = []
        self.input.seek(0)
        for line in self.input:
            if line != '\n':
                word_list.append(cluster_dict[line.split('\t')[1]])
        return word_list
        
    def greater_ave_length(self):
        """Calculates the average length of words in the corpus, then for each
        word checks if it is longer than the average length or not."""
        word_list = []
        self.input.seek(0)
        for line in self.input:
            if line != '\n':
                word_list.append(line.split('\t')[1])
        total = 0
        for word in word_list:
            total += len(word)
        average = total/len(word_list)
        output_list = []
        for word in word_list:
            if len(word) > average:
                output_list.append(True)
            else:
                output_list.append(False)
        return output_list
        
    def is_banana(self):
        """Checks to see if a word is 'banana' or not."""
        word_list = []
        self.input.seek(0)
        for line in self.input:
            if line != '\n':
                if line.split('\t')[1].lower() == 'banana':
                    word_list.append(True)
                else:
                    word_list.append(False)
    
if __name__ == '__main__':
    
    x = Tagger_frame('train.gold')
    x.read()
    x.add_function(x.output_database)
    x.extract()
    #print x.database[0]
    #print x.first_word()
    #print x.brown_cluster(50)
    #print x.greater_ave_length()