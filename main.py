import nltk  
import pickle 
import os 
import numpy 
import re 
from math import log
from time import *
from collections import Counter
 
STOP = 'STOP'
START = '*'
RARE = '_RARE_'
RARE_MAX_FREQ = 5
LOG_OF_ZERO = -1000
PATH = os.path.dirname(os.path.realpath(__file__))

def splitWordTags(sentences):
    tags  = []
    words = []
    for sentence in sentences:
        split_sentence = sentence.split()
        word_tag=[]
        for pair in split_sentence:
            word_tag.append(re.split(r'(^.+)/([A-Z.]+$)', pair)[1:3])
        word_tag = [[START]*2]*2 + word_tag + [[STOP]*2]
        word_tag_trans = numpy.array(word_tag).transpose()
        words.append(list(word_tag_trans[0]))
        tags.append(list(word_tag_trans[1]))
    return words, tags
 
def Qcalc(tags):
    trigrams =[]
    bigrams =[]
    for sentence in tags:
        for trigram in nltk.trigrams(sentence):
            trigrams.append(trigram)
        for bigram in nltk.bigrams(sentence):
            bigrams.append(bigram)
        
    trigramCount = Counter(trigrams)
    bigramCount  = Counter(bigrams)
    Qvalue={}
    for trigram,count in trigramCount.items():
        bgCount = bigramCount[trigram[:-1]]
        Qvalue[trigram] = log(count,2)-log(bgCount,2)
            
    return Qvalue
 
def calculateKnown(words):
    wordsAll =[]
    for sentence in words:
        for word in sentence:
            wordsAll.append(word)
 
    words_count = Counter(word)
 
    knownWordsAll=[]
    for word,count in words_count.items():
        if count > RARE_MAX_FREQ:
            knownWordsAll.append(word)
 
    knownWords = set(knownWordsAll)
 
    return knownWords
 
def replaceRare(sentences, knownWords):
    replacedArray = []
    for sentence in sentences:
        replacedArray.append([word in knownWords and word or RARE for word in sentence])
    return replacedArray
 

def Ecalc(toks, tags):
    tagsFlat = []
    for sentence in tags:
        for word in sentence:
            tagsFlat.append(word)
 
    wordsFlat = []
    for sentence in toks:
        for word in sentence:
            wordsFlat.append(word)
 
    tagsCounter = Counter(tagsFlat)
    wordTag = zip(wordsFlat, tagsFlat)
    wordTagCoungter = Counter(wordTag)
 
    eValue = {}
    for k,c in wordTagCoungter.items():
        firstTagCounter = float(tagsCounter[k[1]])
        eValue[k] = log(float(c),2)-log(firstTagCounter,2)
    tagList = set(tagsFlat)
    return eValue, tagList

def tagVITERBI(tokens, tagset, knownWords, Q, E):
    tags = tagset.difference({START,STOP})
    def S(n):
        if n < 2:
            return [START]
        elif n == T+2:
            return [STOP]
        else:
            return tags
    T = len(tokens)
    pi = [{START: {START: 0.0}}]
    bp = [None]
    for k in range(2,T+2):
        pi.append({})
        bp.append({})
        for u in S(k-1):
            pi[k-1][u] = {}
            bp[k-1][u] = {}
            for v in S(k):
                pi_max = float('-inf')
                w_max = None
                for w in pi[k-2]:
                    q = Q.get((w,u,v),LOG_OF_ZERO)
                    if q == LOG_OF_ZERO:
                        s = q
                    else:
                        p = pi[k-2][w][u]
                        e_word = tokens[k-2]
                        if not e_word in knownWords:
                            e_word = RARE
                        if not (e_word,v) in E:
                            s = LOG_OF_ZERO
                        else:
                            e = E[e_word,v]
                            s = p + q + e
                    if s > pi_max:
                        pi_max = s
                        w_max = w
                if not w_max:
                    continue
                pi[k-1][u][v], bp[k-1][u][v] = pi_max, w_max
    uv = None
    pi_max = float('-inf')
    for u in S(T):
        for v in S(T+1):
            q = Q.get((u,v,STOP),LOG_OF_ZERO)
            if q == LOG_OF_ZERO:
                s = q
            else:
                s = pi[T][u][v] + q
            if s > pi_max:
                pi_max = s
                uv = (u,v)
    y = ['X']*(T+2)
    y[T] = uv[0]
    y[T+1] = uv[1]
    for k in reversed(range(T)):
        y[k] = bp[k+1][y[k+1]][y[k+2]]
    return zip(tokens,y[2:])
 
# Delimiter
DEL = " "
 
DataFolder = os.path.join(PATH, 'data') + '/'
OutputFolder = os.path.join(PATH, 'output') + '/'
ParametersFolder = os.path.join(PATH, 'parameters') + '/'
 

def outputQ(Qvalue, filename):
    outfile = open(filename, "w")
    trigrams = list(Qvalue.keys())
    trigrams.sort()
    for trigram in trigrams:
        line = DEL.join(list(trigram) + [str(Qvalue[trigram])])
        outfile.write(line + '\n')
    outfile.close()
 
def outputE(Evalue, filename):
    outfile = open(filename, "w")
    emissions = list(Evalue.keys())
    emissions.sort()
    for item in emissions:
        output = DEL.join([item[0], item[1], str(Evalue[item])])
        outfile.write(output + '\n')
    outfile.close()
 
def outputTagged(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence + '\n')
    outfile.close()
 
def loadData(name):
    infile = open(DataFolder + name + ".txt", "r")
    data = infile.readlines()
    infile.close()
    return data
 
def saveObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
 
def loadVITERBIParams():
    names = 'tagset knownWords Qvalue Evalue'.split()
    objects = {}
    for name in names:
        with open(ParametersFolder + name + '.pkl', 'rb') as object_file:
            objects[name] = pickle.load(object_file)
 
    return tuple([objects[name] for name in names])
 

def main():
    perf_counter()
 
    train = loadData('Brown_tagged_train')
    words, tags = splitWordTags(train)
    Qvalue = Qcalc(tags)
    saveObject(Qvalue, ParametersFolder + 'q_values.pkl')
    outputQ(Qvalue, OutputFolder + 'q_values.txt')
 
    knownWords = calculateKnown(words)
    saveObject(knownWords, ParametersFolder + 'known_words.pkl')
    wordRare = replaceRare(words, knownWords)
 
    Evalue, tagset = Ecalc(wordRare, tags)
    saveObject(Evalue, ParametersFolder + 'e_values.pkl')
    saveObject(tagset, ParametersFolder + 'tagset.pkl')
    outputE(Evalue, OutputFolder + "e_values.txt")
 
    del train
    del wordRare
 
    dev = loadData('Brown_dev')
    devWords = [] 
    for sentence in dev:
        devWords.append(sentence.split(" ")[:-1])
    VITERBItagged = ( \
        " ".join(["{0}/{1}".format(*x) for x in \
            tagVITERBI(tokens, tagset, knownWords, Qvalue, Evalue)]) \
        for tokens in devWords)
    outputTagged(VITERBItagged, OutputFolder + 'Brown_tagged_dev.txt')
 
 
    print("Elapsed time: " + str(perf_counter()) + ' sec')
 
if __name__ == "__main__": main()
