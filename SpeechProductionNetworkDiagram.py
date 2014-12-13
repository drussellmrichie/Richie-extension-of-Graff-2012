"""
For some reason, this won't run from within canopy. But if you run it from
terminal with "python layeredGraphDrawing2.py" it works fine.

As the network is embiggened, it very quickly gets too crowded for display.
"""

from numpy import genfromtxt
import numpy.random
import random, os #, itertools
import networkx as nx
#from networkx.algorithms import bipartite
import pygraphviz as pgv # pygraphviz should be available

os.chdir('/Users/russellrichie/Google Drive/UConn/Classes/DLC (II)')

# Set some parameters

conceptCount = 3 # do we need more concepts and hence words so that the 
                  # word-form space is a bit more crowded and hence we might find
                  # anti-minimal pair effects?
wordLengths = 2
phonemeSetLength = 9 # setting this to 25 includes all phonemes
nGens = 1000
initAct = 100
eWeights = .3 # I need to read Martin more closely to see what exactly he set his weights to. He seems to set them to different values for dif sims
nWeights = 1 # eWeights is edge weights, which in footnote 11 Martin says he set to .3 for all connections. nWeight is node weight, which I am leaving at 1
decay = 0.6
settleTime = 10 # how many 'round-trips' in the network does this allow? Martin allowed
                # activation to spread for 5 time-steps, which is reportedly enough time
                # for type frequency effects in Dell and Gordon (2003) to affect lexical
                # nodes

phonemes = []
distFeatData = genfromtxt('english distinctive features.csv', delimiter=',',dtype=str)
for row in distFeatData[1:phonemeSetLength + 1]:
    phonemes.append(row[0])

def sumInputs(weights, inputs):
    return sum(weights[index]*inputx for index, inputx in enumerate(inputs))

def newActivation(weightj, prevAct, netInputs, decay):
    return weightj*(prevAct + netInputs)(1-decay) + numpy.random.normal(scale=.05*prevAct)

def initWords(phonemes,conceptCount,wordLengths):
    newWords = []
    for _ in range(conceptCount):
        newWords.append(frozenset(random.sample(phonemes, wordLengths))) # must be frozen (hashable) so networkx can make nodes out of words
    #for _ in range(wordCount):
    #    newWords.append([random.choice(phonemes) for _ in range(wordLengths)])
    return newWords

wordList = initWords(phonemes,conceptCount,wordLengths)

def coinOneSyn(phonemes,wordLengths):
    return frozenset(random.sample(phonemes, wordLengths))

# make network!

phGraph0 = nx.Graph()
featureList = list(distFeatData[0][1:])
phGraph0.add_nodes_from(featureList, activation=0, rank=0)                    # add feature layer, eventually rank attribute will 
                                                                                            #     allow drawing network into proper layers
phGraph0.add_nodes_from(phonemes, activation=0, rank=1)                                     # add phoneme layer
phGraph0.add_nodes_from(wordList, activation=0, rank=2)  # add word layer
phGraph0.add_nodes_from(range(conceptCount), activation=0, rank=3)  # add concept layer
for wordInd, word in enumerate(wordList):
    phGraph0.add_edge(word, wordInd, weight = eWeights)                                     # add edges between concepts and words
    for phoneme in word:
        phGraph0.add_edge(phoneme, word, weight = eWeights)                                 # add edges between words and phonemes
    #for wordInd2, word2 in enumerate(wordList[wordInd:]):                                   # I think indexing by wordInd should fix extra link problem
    for wordInd2, word2 in enumerate(wordList):                                             # I think indexing by wordInd should fix extra link problem
        if word2 != word:                             # another, less efficient way of checking if edge already exists
        #if not phGraph0.has_edge(word,word2) and word2 != word:                             # another, less efficient way of checking if edge already exists
            phGraph0.add_edge(word, word2, weight = -eWeights)   # what should weight be?; add edges between words (lateral inhibition!)
for phoind, phoneme in enumerate(phonemes):                                                 # add edges between phonemes and features
    for featind, feature in enumerate(distFeatData[0][1:]):
        if distFeatData[phoind + 1][featind + 1] == '+': # must add 1 to each to skip the blank first cell of distFeatData
            phGraph0.add_edge(phoneme,feature, weight = eWeights)

# now remove any (feature) nodes without any neighbors...maybe there's a better way...
for feature in featureList:
    if not phGraph0[feature]:
        phGraph0.remove_node(feature)

A = nx.to_agraph(phGraph0)
three = A.add_subgraph(range(conceptCount),rank='same')
two = A.add_subgraph(wordList,rank='same')
one = A.add_subgraph(phonemes,rank='same')
zero = A.add_subgraph(list(distFeatData[0][1:]),rank='same')
A.draw('network architecture.png', prog='dot')