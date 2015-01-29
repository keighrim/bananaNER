import collections

class Tagger_frame():
    
    def __init__(self,file_name):
        self.inpath = '../dataset/'+file_name
        self.outpath = '../result/output.txt'
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
    
if __name__ == '__main__':
    
    x = Tagger_frame('train.gold')
    x.read()
    x.add_function(x.output_database)
    x.extract()
    