
# coding: utf-8

# In[2]:


import pandas as pd
import re
import os.path
from collections import Counter,defaultdict
from nltk.stem import PorterStemmer
import numpy as np
import math
import os
import codecs
import nltk
import sys


# In[3]:


class Indexer:
    def __init__(self): 
        self.universallist = []      #List to store the complete list of words in the corpus
        self.index = defaultdict(dict)    #dictionary to store the term frequency for each doc and to store document frequency 
        self.doctermfreq = defaultdict(dict) #contains docname followed by the term and its frequency 
        self.docvector = defaultdict(dict) #to store the document vector
        self.weight = defaultdict(list)    #to store the w^2
        
    
    def tokenize(self,text):
        token = [] # main list of tokens
        rule1 = re.findall(r'\b(\w+[-]\n\w+)\b',text) #to find words ending with - and a line break followed by another word
        token.extend([re.sub(r'[-]\n','',i) for i in rule1])#replacing the - with nothing 
        token.extend(re.findall(r'\b(\w+@[\w\.]+(?:.com|.edu))\b',text))
        token.extend(re.findall(r'\b(\d+[.]\d+[.]\d+[.]\d+)\b',text))
        token.extend(re.findall(r'\b((?:https:\/\/)?www[.]\w+[.\w]+)\b',text))
        token.extend(re.findall(r"(?<=[\s,])'(.*?)\.?'(?=[\s,.])", text))
        token.extend(re.findall(r"(?:^|[\s,.\"])[A-Z][a-z']+?(?: [A-Z][a-z']+)+(?=[\s\",!.]|$)",text))
        token.extend(re.findall(r'[A-Z][.](?:[A-Z][.]?)+',text))
        

        #replacing all the tokens found using the previous 5 rules with empty 

        modfilelist = re.sub(r'\b(\w+[-]\n\w+)\b','',text)
        modfilelist = re.sub(r'\b(\w+@[\w\.]+(?:.com|.edu))\b','',modfilelist)
        modfilelist = re.sub(r'\b(\d+[.]\d+[.]\d+[.]\d+)\b','',modfilelist)
        modfilelist = re.sub(r'\b((?:https:\/\/)?www[.]\w+[.\w]+)\b','',modfilelist)
        modfilelist = re.sub(r"(?<=[\s,])'(.*?)\.?'(?=[\s,.])",'',modfilelist)
        modfilelist = re.sub(r"(?:^|[\s,.\"])[A-Z][a-z']+?(?: [A-Z][a-z']+)+(?=[\s\",!.]|$)",'',modfilelist)
        modfilelist = re.sub(r'[A-Z][.](?:[A-Z][.]?)+','',modfilelist)

        #replacing --,_,*,=,(,) with white space 
        modfilelist = re.sub(r'--',' ',modfilelist)

        modfilelist = re.sub(r'_',' ',modfilelist)

        modfilelist = re.sub(r'\*',' ',modfilelist)

        modfilelist = re.sub(r'=',' ',modfilelist)

        modfilelist = re.sub(r'\(','',modfilelist)

        modfilelist = re.sub(r'\)','',modfilelist)




        #Finally tokenizing the remaining tokens found in the text
        token.extend(re.findall(r'(?:^|[\s,.\'])\"?([\w\-]+)\"?(?=[\s,.!\']|$)',modfilelist))

        newtoken = [re.sub(r'\.','',x) for x in token]      #removing . in acronyms 
        newtoken = [re.sub(r'[,]','',x) for x in token]     #removing , 
        #to filter out the numbers
        return newtoken

    def preprocessing(self,text,stopwords):
        
        stemmer = PorterStemmer() 
        #removing the ones present in stop list and length less than 3 
        newlist = [x for x in text if x.lower() not in stopwords]
        newlistmod = [x for x in newlist if len(x) > 2]
        stemmed = [stemmer.stem(w) for w in newlistmod]     #stemming the words
        
        uni = [x.lower() for x in newlistmod]               #storing the words after making them lower for spelling correctiom
        self.universallist.extend(uni)

        
        return stemmed
    
    def termfrequency(self, name,words):
        count = Counter(words)     #calling counter to find the number of times a word appears in a document 

        for k, i in count.items():
            self.index[k][name] = i      #term frequency for each term in the document    

            self.doctermfreq[name][k] = i   #contains docname followed by the term and its frequency 
                  
                
    def docfrequency(self):
        terms = self.index.keys()        #copying the vocabulary terms
        
        for t in terms:
            dfreq = len(self.index[t])    #finding the number of documents each term contain
            self.index[t]['invdocfreq'] = math.log(N/(dfreq + 1)) #calculating the inverse document frequency
            
            
    
    def writeindex(self):
        #method to write the index to a file 
        n = ['invdocfreq']
        with codecs.open('index.txt', 'w','utf-8') as fo:
            for term,value in self.index.items():
                temp = [term]
                for docname, tf in value.items():
                    if docname not in n:
                        docname = re.sub(r'[,]','',docname)
                        temp.append(docname)
                        temp.append(tf)
                    else:
                        temp.append(tf)
                line = ','.join([str(n) for n in temp])
                fo.write(line + os.linesep)
    
    def readindex(self):
        #method to read the index and store it in a dictionary format
        with codecs.open('index.txt','r',encoding='utf-8') as f:
            text = f.readlines()
            for t in text:
                temp = t.split(',')
                temp = [t.strip() for t in temp]
                #print(temp)
                token = temp[0]
                self.index[token]['invdocfreq'] = float(temp[-1])
                del temp[-1]
                i = 1
                while i<len(temp):
                    self.index[token][temp[i]] = int(temp[i+1])
                    i +=2
                    
    def docvectorize(self):
        tokens = self.index.keys()    #storing the vocabulary term in a variable
    
        notdocname = ['invdocfreq']

        for t in tokens:
            for docname, termfreq in self.index[t].items():
                if docname not in notdocname:
                    tfidf = termfreq * self.index[t]['invdocfreq']  #calculating the tfidf for each term 
                    self.docvector[docname][t] = tfidf #storing the document name and its terms followed by the tfidf

        docnames = self.docvector.keys()
        # to calculate the w^2 of the cosine similarity formula
        for d in docnames:
            for term, tfidf in self.docvector[d].items():
                self.weight[d].append(tfidf*tfidf)
                
                
                
                
    def queryweight(self,countertoken):
        tokens = countertoken.keys()      #getting the vocabulary terms 
        queryvector = defaultdict(dict)  #query vector
        for t in tokens: #for each vocabulary term 
            invdocfreq = [self.index[t]['invdocfreq'] if t in self.index else 0][0]  # if the term is present in the query then it invdocfreq or 0 otherwise
            #qw = idf * tf
            queryvector[t] = invdocfreq * countertoken[t] #calculating the query vector 
        return queryvector
    
    def cosinesimiliarity(self,querytoken,queryvector): 
        q = []  # to hold q^2 in the bottom of cosine similarity formula
        doc = dict()
        for t in querytoken: # for each term in the query we calculate the Q^2
            #q^2
            q.append(queryvector[t]*queryvector[t])

        term = self.index.keys()   
        docname = self.docvector.keys()
        if sum(q) != 0:
            for d in docname:
                wq = []
                for t in term: # for each term in the index
                    tfidf = [self.docvector[d][t] if  t in self.docvector[d] else 0][0]  #obtaining the tfidf if the term is present in the document else storing a 0
                    queryweight = [queryvector[t] if t in queryvector else 0][0] #getting the query weight for that term 
                    wq.append(tfidf*queryweight) # caculating the wq on the numerator of the cosine similarity formula

                w = self.weight[d]
                #print(sum(w))
                cosinesimilarity = sum(wq) / ((math.sqrt(sum(w))) * (math.sqrt(sum(q))))
                doc[d] = cosinesimilarity
            return doc   
        else:
            return 0
        
        
    def rocchio(self,r,nr,oldquery):
        ALPHA = 0
        BETA = 0.75
        GAMMA = 0.25

        token = self.index.keys()
        reldoc = {}
        nonreldoc = {}
        query = {}

        q = {}

        for t in token:
            for d in r:
                if t in reldoc:
                    #calculating the sum of the relevant document vectors
                    reldoc[t] = reldoc[t] + [self.docvector[d][t] if  t in self.docvector[d] else 0][0]
                else:
                    reldoc[t] = [self.docvector[d][t] if  t in self.docvector[d] else 0][0]

            for d in nr:
                if t in nonreldoc:
                    #calculating the sum of irrelevant document vectors 
                    nonreldoc[t] = nonreldoc[t] + [self.docvector[d][t] if  t in self.docvector[d] else 0][0]
                else:
                    nonreldoc[t] = [self.docvector[d][t] if  t in self.docvector[d] else 0][0]
            if t in oldquery:
                #constructing the query vector 
                q[t] = oldquery[t]
            else:
                q[t] = 0

        for t in token:
            #applying the rocchio algorithm to construct the new query vector 
            query[t] = ALPHA*q[t] + BETA*(reldoc[t]/len(r)) - GAMMA*(nonreldoc[t]/len(nr))

            
        for t in token:
            #replacing the negative value in the new query vector with 0 
            query[t] = [query[t] if query[t] >= 0 else 0.0][0]
        return query
    
    
    def editdistance(self,query_term,term):
        #edit distance function
        ed=[[0 for x in range(len(query_term)+1)] for x in range(1+len(term))] # creating a matrix of 0s and applying the edit distance algorithm
        #print(dp)
        for i in range(1+len(query_term)):
            ed[0][i]=i
        for i in range(1+len(term)):
            ed[i][0]=i
        for i in range(1,1+len(term)):
            for j in range(1,1+len(query_term)):
                if term[i-1]==query_term[j-1]:
                    ed[i][j]=ed[i-1][j-1]
                else:
                    ed[i][j]=1+min(ed[i-1][j],ed[i][j-1],ed[i-1][j-1])
        return ed[len(term)][len(query_term)]
    
    def bigram(self,term):  

        # Insert a space inbetween each character in myString

        spaced = ''
        for ch in term:
            spaced = spaced + ch + ' '

        # Generate bigrams out of the new spaced string

        tokenized = spaced.split(" ")
        myList = list(nltk.bigrams(tokenized))
        return myList
    
    def jaccard(self,query,vocab): 
        #calculating the jacard coefficient |A and B|/|A or B|
        q = self.bigram(query)
        termlist = {}
        #calculating how many bigrams between the query and the terms in the index match
        for t in vocab:
            voc = self.bigram(t)
            count = 0
            for i in range(len(list(q))):
                for j in range(len(voc)):
                    if list(q)[i] == list(voc)[j]:
                        count += 1
            
            jc = count/(len(list(q)) + len(list(voc)) - count)
            # selecting only terms with jacard coefficient is greater than 0.1
            if jc > 0.1:
                termlist[t] = jc

        return termlist 


    
    def spell_check(self,word,vocab):
        final = {}
        #calculating the jacard coefficient between the query term and all the index terms and returning it in a dictionary 
        termlist = self.jaccard(word,vocab)
        i=0
        # calculating the editdistance between the terms and query 
        for term, bigram in termlist.items():
            final[term] = self.editdistance(word,term)
            
        min = 100000
        # finding the min edit distance from the list 
        for x in final.keys():
            word_check=0
            if abs(len(word)-len(x))<3:
                word_check=1
            if final[x]<min and word_check==1: 
                #index=x
                min=final[x]
        idx=[]
        
        correct_words=[]
        # if the minimum distance is 0 then the query term has been found
        if min==0:
            correct_words.append(word)
        else:
            #else the terms with that min value is returned
            for x in final.keys():
                if final[x]==min:
                    correct_words.append(x)
            
        return correct_words
                
        
    
    def spelling(self,query):
        universal = set(self.universallist)
        terms = list(universal) # getting a list of all words from the corpus excluding the stop words and without lemmetisation 
        words=re.compile('\w+').findall(query.lower()) #splitting the query into individual query terms
        words = [x for x in words if x.lower() not in stopwords] #removing the stopwords
        new_word=[]
        #for each word in the query 
        for wrd in words:
            #we get the probable correct spellings
            probables = self.spell_check(wrd,terms)
            # if there was only one of word in the list it means its the query term
            if len(probables)==1 and probables[0]==wrd:
                new_word.append(wrd)
            else:
                #the query term wasnt found in the corpus
                print("The word wasnt present the vocabulary\n")
                
                print(probables) 
                while 1:
                    #getting input from the user in the form of position number for the correct word 
                    a=input("which one did you mean? Choose the position of the word starting from 1 \n")
                    # checking if the user entered the position number 
                    if( a.isdigit()):
                        new_word.append(probables[int(a)-1])
                        break
                    else:
                        print("User input is string ")
                        
                       
                 
        # the correct spellings in the list are combined together into a string and returned        
        cquery=' '.join(word for word in new_word)
        return cquery


    
    
          
    
    
    
    
    
    
                
                
    
    
        
    
    


# In[4]:


if __name__ == '__main__':
    arglist = sys.argv
    if len(arglist) < 2:
        print("Usage: <file path> <stopword path along with the stopword file name>")
        sys.exit(1) #exit interpreter
    #getting the path of the files
    file_path = arglist[1]
    #opening the stopword list from the specified path 
    stopword = open(arglist[2],mode = "r",encoding="utf-8-sig")
    stopwords = stopword.readlines()
    stopword.close()
    #removing the \n from the stopwords
    stopwords = [re.sub(r'\n','',x) for x in stopwords] 
    #creating the indexer class object
    index = Indexer()
    
    N = 0                                                         #no of documents 
    #indexing part
    for xfile in os.listdir(file_path):
        N += 1
        filename = re.findall(r'(.+?).txt',xfile)
        xfile = os.path.join(file_path, xfile)
        file = open(xfile,mode = "r",encoding="utf-8-sig")
        filelist = file.read()
        file.close()
        tokenlist = index.tokenize(filelist)                    #tokenizing the file
        processeddoc = index.preprocessing(tokenlist,stopwords)  #preprocessing the tokens
        #storing the termfrequency for each document 
        index.termfrequency(filename[0],processeddoc) # document frequency

    index.docfrequency()   #for calculating document frequency and IDF

    index.writeindex()    #for writing the index
    index.docvectorize()    #calculating the document vectors
    
    #querying part 
    userinput = ''
    while userinput != 'q':                                         #checking whether the user wants to quit

        queryvector = defaultdict(list)                             #To store the query vector
        userinput = input("Enter the query or type q to exit ")     #getting the query from the user
        if userinput == 'q':                                        # if the input is q then it quits
            break
        correctquery = index.spelling(userinput)                    #checking the spelling of the query terms

        querytoken = index.tokenize(correctquery)                   #tokenizing the correct query
        processedquery = index.preprocessing(querytoken,stopwords)  #removing stopwords and stemming the query terms
        querytermfreq = Counter(processedquery)                     #term frequency of the query terms
        queryvector = index.queryweight(querytermfreq)              #calculating query vector
        output = ''
        while output != 'q':
            result = index.cosinesimiliarity(processedquery,queryvector)  #calculating the cosine similarity
            if result != 0:                                               #ensuring the result isnt 0 which means query didnt match any document
                sorted_by_value = sorted(result.items(), key=lambda kv: kv[1],reverse=True)    #sorting the documents in the decreasing order of the cosine similarity 
                counter = 0
                precisionatK = 0
                r = [] #To store relevant documents
                nr = [] #To store non relevant documents
                for item in sorted_by_value:
                    if counter <= 15:
                
                        counter += 1
                        print("The" + ' '+ str(counter) + ' ' + "result" + ' ' + str(item[0]))
                        while 1:
                            #getting feedback from the user
                            feedback = input("\nType Y/YES if relevant or type N/NO if irrelevant?")
                            if (feedback.upper() =='Y')|(feedback.upper() =='YES'):
                                precisionatK +=1
                                r.append(item[0])
                                break

                            elif (feedback.upper() =='N')|(feedback.upper() =='NO'):
                                nr.append(item[0])
                                break

                            else:
                                print("Please enter Y/y or N/n only!")
                                continue 
                #checking the precision 
                precision = precisionatK/counter
                #if the precision is 0 it means all returned documents are irrelevant
                if precision == 0:
                    print("All the documents returned is irrelevant cant retrieve anymore for this query")
                    break
                #precision of 1 means all the documents displayed is relevant 
                if precision == 1:
                    print("All the document displayed is relevant")
                    break
                #calculating the new query vector with the rocchio algorithm
                queryvector = index.rocchio(r,nr,queryvector)
                output =  input("If you want to quit type q or press anything else to see the new results ")

            else:
                print("Query term didnt not match with any document ")
                break


# # reference 
# https://github.com/rohanag/ADB-relevance-feedback-rocchio
# 
# https://github.com/aimannajjar/columbiau-rocchio-search-query-expander
# 
# https://github.com/harora/Spell-Checker/blob/master/index.py
