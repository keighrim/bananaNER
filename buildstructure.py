import collections
import urllib2
import xml.etree.cElementTree as ET

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
                
    def make_feature_file(self):
        '''offer token for feature function in feature functions(where many feature 
        functions in extraction), then extraction functions return the values of all
        the functions which is appended to the append_content.Finally, write to the
        output file
        '''
        self.add_function(self.initcap)
        self.add_function(self.onecap)
        self.add_function(self.allcap_period)
        self.add_function(self.contain_digit)
        self.add_function(self.twod)
        self.add_function(self.fourd)
        self.add_function(self.digit_slash)
        self.add_function(self.dollar)
        self.add_function(self.percent)
        self.add_function(self.digit_period)
        
        input = open(self.inpath)
        output_feature_file=open("../feature_output.txt",'w')
        for line in input:
            if line == '\n':
                continue
            list=line.split()
            token=list[1]
            append_content=self.extract(token)
            output_feature_file.write(line[0:-1]+append_content+'\n')
                        
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
        
    def extract(self,token):
        '''traverse and execute the list of functions'''
        result=''
        for function in self.feature_functions:
            result+=' '+function(token)
        return result
    
    def initcap(self,token):
        if token[0].isupper() and token[-1] == '.':
            return 'initcap=True'
        else: return 'initcap=False'
            
    def onecap(self,token):
        if token[0].isupper() and len(token) == 1:
            return 'onecap=True'
        else: return 'onecap=False'
    
    def allcap_period(self,token):
        for letter in token[0:-1]:
            if not letter.isupper():
                return 'allcap_period=False'
        if token[-1]=='.':
            return 'allcap_period=True'
        else: return 'allcap_period=False'
        
    def contain_digit(self,token):
        for letter in token:
            if letter.isdigit():
                return 'contain_digit=True'
        return 'contain_digit=False'
        
    def twod(self,token):
        if len(token)==2:
            for letter in token:
                not letter.isdigit()
                return 'twod=False'
            return 'twod=True'
        else: return 'twod=False'
        
    def fourd(self,token):
        if len(token)==4:
            for letter in token:
                if not letter.isdigit():
                    return 'fourd=False'
            return'fourd=True'
        else: return 'fourd=False'
    
    def digit_slash(self,token):
        for letter in token:
            if letter != '//' and not letter.isdigit():
                return 'digit_slash=False'
        return 'digit_slash=True'
    
    def dollar(self,token):
        for letter in token:
            if letter == '$':
                return 'dollar=True'
        return 'dollar=False'
    
    def percent(self,token):
        if '%' in token:
            return 'percent=True'
        else: return 'percent=False'
        
    def digit_period(self,token):
        for letter in token:
            if letter.isdigit():
                if '.' in token:
                    return 'digit_period=True'
        return 'digit_period=False'

'''    
    def build_loc_dic(self):
        file = urllib2.urlopen('http://www.timeanddate.com/')
        data = file.read()
        file.close()
        tree=ET.parse(file)
        root=tree.getroot()
        for tags in root[1]:
            print tags
'''        

        
                                         
'''
    def tokenbased_feature(self,token):
        first_letter_cap=False
        last_letter_period=False
        cap_num=0
        digit_num=0
        contain_dollar=False
        contain_slash=False
        contain_percent=False
        contain_period=False
        token_length=len(token)
        if token[0].isupper(): first_letter_cap=True
        if token[-1]=='.': last_letter_period=True
        
        for letter in token:
            if letter.isupper():
                cap_num+=1
            elif letter.isdigit():
                digit_num+=1
            elif letter=='//':
                contain_slash=True
            elif letter=='.':
                contain_period=True
            elif letter=='$':
                contain_dollar=True
            elif letter=='%':
                contain_percent=True
        
        if first_letter_cap and last_letter_period:
            return 'InitCap-Period'
        elif cap_num==token_length and cap_num==1:
            return 'One Cap'
        elif cap_num==token_length-1 and last_letter_period:
            return 'AllCap-Period'
        elif digit_num>=1:
            return 'Contain-Digit'
        elif digit_num==2 and token_length==2:
            return 'TwoD'
        elif digit_num==4 and token_length==4:
            return 'FourD'
        elif digit_num>=1 and contain_slash:
            return 'Digit-slash'
        elif contain_percent:
            return 'Percent'
        elif contain_dollar:
            return 'Dollar'
        elif contain_period and digit_num>=1:
            return 'Digit-Period'
        else:
            return ""
'''
           
        
if __name__ == '__main__':
    
    x = Tagger_frame('train.gold')
    x.read()
    #x.add_function(x.output_database)
    #x.extract()
    #x.tokenbased_feature('')
    #x.make_feature_file()
    x.build_loc_dic()
    print 'finished'
    
    