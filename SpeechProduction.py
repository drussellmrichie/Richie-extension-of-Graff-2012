# -*- coding: utf-8 -*-

from __future__ import division

"""
REIMPLEMENTATION OF MARTIN (2007) / GRAFF (2012) SPREADING ACTIVATION MODEL OF
SPEECH PRODUCTION

The goal is to replicate Graff (2012)'s finding that more pereptible contrasts
have more minimal pairs.
"""

from numpy import genfromtxt
import numpy as np
import numpy.random
from numpy import array
import random, os #, itertools
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import images2gif

os.chdir('/Users/russellrichie/Richie extension of Graff (2012)/Richie-extension-of-Graff-2012')

# Set some parameters

class prettyfrozenset(frozenset):
    def __str__(self):
        initial = super(prettyfrozenset, self).__str__()
        return initial[initial.index("(") + 1:-1]

conceptCount = 10 # do we need more concepts and hence words so that the 
                  # word-form space is a bit more crowded and hence we might find
                  # anti-minimal pair effects?                  

wordLengths = 3   # the number of phonemes in a word. problem with setting this
                  # low (and making minimal pairs more likely) is that it makes 
                  # exact homophones likely. If a homphone is deleted from the
                  # lexicon, then it leaves another concept without any word to
                  # express it, and the simulation crashes.

phonemeSetLength = 8 # setting this to 25 includes all phonemes
                      # note that because I've outlawed any homophones,
                      # conceptCount must < phonemeSetLength choose word lengths

nGens = 2       # the number of generations to be simulated

initAct = 100   # the initial activation of the concept to be expressed

eWeights = .3   # I need to read Martin more closely to see what exactly he set 
                # his weights to. He seems to set them to different values for 
                # dif sims
                
nWeights = 1    # eWeights is edge weights, which in footnote 11 Martin says he 
                # set to .3 for all connections. nWeight is node weight, which 
                # I am leaving at 1, also consistent with Martin on page 35
decay = 0.6     # decay rate

settleTime = 10  # how many 'round-trips' in the network does this allow? Martin allowed
                # activation to spread for 5 time-steps, which is reportedly enough time
                # for type frequency effects in Dell and Gordon (2003) to affect lexical
                # nodes

# Define some functions

def sumInputs(weights, inputs):
    return sum(weights[index]*inputx for index, inputx in enumerate(inputs))

def newActivation(weightj, prevAct, netInputs, decay):
    return weightj*(prevAct + netInputs)*(1-decay) + numpy.random.normal(scale=.05*abs(prevAct))

def initWords(phonemes,conceptCount,wordLengths):
    newWords = []
    for _ in range(conceptCount):
        newWords.append(prettyfrozenset(random.sample(phonemes, wordLengths))) # must be frozen (hashable) so networkx can make nodes out of words
    #for _ in range(wordCount):
    #    newWords.append([random.choice(phonemes) for _ in range(wordLengths)])
    return newWords

def coinOneSyn(phonemes,wordLengths):
    return prettyfrozenset(random.sample(phonemes, wordLengths))

# initialize the phonemes and word list

phonemes = []
distFeatData = genfromtxt('english distinctive features.csv', delimiter=',',dtype=str)
for row in distFeatData[1:phonemeSetLength + 1]:
    phonemes.append(row[0])
    
wordList = initWords(phonemes,conceptCount,wordLengths)

# make network!

phGraph0 = nx.Graph()
phGraph0.add_nodes_from(list(distFeatData[0][1:]), activation=0.01, rank=0)                 # add feature layer, eventually rank attribute will 
                                                                                            # allow drawing network into proper layers
phGraph0.add_nodes_from(phonemes, activation=0.01, rank=1)                                  # add phoneme layer
phGraph0.add_nodes_from(wordList, activation=0.01, rank=2)                                  # add word layer
phGraph0.add_nodes_from(range(conceptCount), activation=0.01, rank=3)                       # add concept layer
for wordInd, word in enumerate(wordList):
    phGraph0.add_edge(word, wordInd, weight = eWeights)                                     # add edges between concepts and words
    for phoneme in word:
        phGraph0.add_edge(phoneme, word, weight = eWeights)                                 # add edges between words and phonemes
    for wordInd2, word2 in enumerate(wordList):
        if word2 != word:                                                                   # so this prevents words from laterally inhibitting themselves, but also
                                                                                            # somehow prevents other double-linkages between words from appearing??
            phGraph0.add_edge(word, word2, weight = -eWeights)                              # add edges between words (lateral inhibition!)
for phoind, phoneme in enumerate(phonemes):                                                 # add edges between phonemes and features
    for featind, feature in enumerate(distFeatData[0][1:]):
        if distFeatData[phoind + 1][featind + 1] == '+':                                    # must add 1 to each to skip the blank first cell of distFeatData
            phGraph0.add_edge(phoneme,feature, weight = eWeights)

# phGraph1 = phGraph0.copy() # need one network for time t, and another for time t-1...maybe there's a way around this

diffList = [] # this will keep track of the average similarity among words 
              # at each generation

# change dir so network activation diagrams are saved in the right place
os.chdir('/Users/russellrichie/Richie extension of Graff (2012)/Richie-extension-of-Graff-2012/network activation diagrams')

# creating layered positioning for nodes...sure feels like a lot of work, but
# i have no version of python that plays nice with both pygraphviz and matplotlib!!!

featureY = 0
phonemeY = 2
wordY    = [3.5, 4.5] * len(wordList) # this will be longer than it needs to be
conceptY = 6

featureLen = len(distFeatData[0][1:])
featureX  = range( int(-featureLen/2)    + 0, int(featureLen/2)    + 2 )
phonemeX  = range( int(-len(phonemes)/2) + 1, int(len(phonemes)/2) + 1 )
wordX     = range( int(-len(wordList)/2) + 1, int(len(wordList)/2) + 1 )
conceptX  = range( int(-conceptCount/2)  + 1, int(conceptCount/2)  + 1 )

pos = dict()
for node in phGraph0:
    if type(node)    == prettyfrozenset:
            pos[node] = array([wordX.pop()    * 4, wordY.pop()])
    elif type(node)  == int:
            pos[node] = array([conceptX.pop() * 3, conceptY])
    elif node in distFeatData[0][1:]:
            pos[node] = array([featureX.pop() * 3, featureY])
    else:
            pos[node] = array([phonemeX.pop() * 3, phonemeY])

#nx.draw_networkx(phGraph0,
#            pos,
#            node_size= 1000,
#            cmap = 'hot',
#            alph =0.8)                                                                                                                                                                                    

# simulate!!!
for currGen in range(nGens):
    for currCon in range(conceptCount):
        print "Current generation is {}".format(currGen)
        print "Current concept is {}".format(currCon)

        # coin a new synonym and link it to its concept and to its phonemes
        newsyn = coinOneSyn(phonemes, wordLengths)
        oldsyn = phGraph0.neighbors(currCon)[0]
        while newsyn in wordList:                                             # this ensures that the newsyn is not the same as the oldsyn  
            print "newsyn still in wordList!"                                 # or any other word in the lexicon, as this will crash the sims
            newsyn = coinOneSyn(phonemes, wordLengths)            
        phGraph0.add_node(newsyn, activation=0.01, rank=2)
        phGraph0.add_edge(newsyn, currCon, weight = eWeights)                 # link syn to its concept
        wordList = [n for n in phGraph0 if phGraph0.node[n]['rank']==2]       # generator of words in network!
        wordList.remove(newsyn)                                               # remove b/c don't want to link word to itself
        for word in wordList:
            phGraph0.add_edge(word, newsyn, weight = -eWeights)               # link syn to other words
        for phoneme in newsyn:
            phGraph0.add_edge(newsyn, phoneme, weight = eWeights)             # link syn to its phonemes
        wordList.append(newsyn)                                               # put newsyn back into wordlist because we'll need it later
        
        phGraph1 = phGraph0.copy() # need one network for time t, and another for time t-1...maybe there's a way around this

        phGraph0.node[currCon]['activation'] = 100              # set activation of concept to 100, and we're off!
        for currStep in range(settleTime):                      # allow activation to flow for settleTime # of steps
            for currNode in phGraph1.nodes(): #if type(node) != int:# update activation for every node in network...not sure if I should exclude concept nodes?
                neighborInputs = [phGraph0.node[neighbor]['activation'] for neighbor in phGraph0.neighbors(currNode)]
                curreWeights = [phGraph0[currNode][neighbor]['weight'] for neighbor in phGraph0.neighbors(currNode)]
                netInputs = sumInputs(curreWeights, neighborInputs)    # how do I deal with different weights, esp between words???
                phGraph1.node[currNode]['activation'] = newActivation(nWeights, phGraph0.node[currNode]['activation'], netInputs, decay)   # how do I deal with different weights, esp between words???
            phGraph0 = phGraph1.copy() # and with the end of this timestep, time 1 is now time 0!
            
            
            # Plot the network and its nodes' activations at each time step
            
            activations = [0] * len(phGraph0)
            for index, currNode in enumerate(phGraph0):
                activations[index] = phGraph0.node[currNode]['activation']
            
            #print activations
            
            wordX     = range( int(-len(wordList)/2), int(len(wordList)/2) + 2 )
            wordY     = [3.5, 4.5] * len(wordList) # this will be longer than it needs to be

            for node in phGraph0:
                if type(node)    == prettyfrozenset:
                        pos[node] = array([wordX.pop() * 4, wordY.pop()])
            """
            nx.draw_networkx(phGraph0,
                        pos,
                        node_color = activations,
                        node_size= 1000,
                        cmap = 'YlOrRd',
                        alpha =0.9)                                                                                                                                                                                    
            """
            plt.clf() # in case it wasn't already closed before??
            
            # super clunky code to extract target nodes (concepts down to features)
            # and then change node_sizes to make those nodes bigger
            
            nodes = phGraph0.nodes()
            specNodes = [currCon]
            for specNode1 in phGraph0.neighbors(currCon):
                specNodes.append(specNode1)
                for specNode2 in phGraph0.neighbors(specNode1):
                    if phGraph0.node[specNode2]['rank'] == 1:
                        specNodes.append(specNode2)
                        for specNode3 in phGraph0.neighbors(specNode2):
                            if phGraph0.node[specNode3]['rank'] == 0:
                                specNodes.append(specNode3)
            specNodes = set(specNodes)
            node_sizes = [1000] * len(nodes)
            for index, node in enumerate(nodes):
                if node in specNodes:
                    node_sizes[index] = 2000
            
            nx.draw_networkx_nodes(phGraph0,
                       pos,
                       #nodelist = nodes,
                       node_color= activations, # not sure if I have to convert activations to points on some color scale
                       node_size=node_sizes,
                       cmap = 'YlOrRd',
                       vmin = -10,
                       vmax = 10,
                       alpha=0.9)
            nx.draw_networkx_labels(phGraph0,
                       pos)
                       
            edges = phGraph0.edges()
            pltEdges = []
            for edge in edges:
                if type(edge[0]) != prettyfrozenset or type(edge[1]) != prettyfrozenset:
                    pltEdges.append(edge)
                       
            nx.draw_networkx_edges(phGraph0,
                       pos,
                       edgelist=pltEdges,
                       alpha = .3)
         
            plt.text(-28,2,"currStep: {}".format(currStep), fontsize = 30)
            plt.text(-28,1,"currGen: {}".format(currGen), fontsize = 30)
                                                                                                                  
            filename = 'net acts gen= {} con= {} t= {} with background.png'.format(currGen, currCon, currStep)
            fig = plt.gcf()
            fig.set_size_inches(18.5,10.5)
            #plt.axis('off') # comment this off for making gifs/animations, keep it on for regular images for ppt, etc.
            plt.savefig(filename,bbox_inches='tight')
            plt.clf() # close the figure so we don't keep writing on the same fig every iter of loop!
            
        # Let's see if the right word(s) were activated....
                
        print "Target word was {} or {}".format(phGraph1.neighbors(currCon)[0],phGraph1.neighbors(currCon)[1])
        #print "Target word was {}".format(phGraph1.neighbors(currCon)[0])
        #wordList.append(newsyn) # put the word back in
        wordList = sorted(wordList, key=lambda x: phGraph1.node[x]['activation'], reverse = True)
        for x in wordList:
            if x == phGraph1.neighbors(currCon)[0] or x == phGraph1.neighbors(currCon)[1]:
                print "    " + str(x) + ": " + str(phGraph1.node[x]['activation']) + "   <---- target"
            else:
                print "    " + str(x) + ": " + str(phGraph1.node[x]['activation'])
                
        # remove the synonym with less activation...uncomment the if/thens to 
        # avoid removing homophononous words from their other concepts! only 
        # need this code if you *didn't* prevent homophones from being coined 
        # earlier. The code is very inelegant!
    
        #otherCons = [x for x in range(conceptCount) if x != currCon]
                                                                                                                                                                                                                                                                                                            
        if phGraph1.node[newsyn]['activation'] > phGraph1.node[oldsyn]['activation']:
            print "    Dumping oldsyn"
            phGraph1.remove_node(oldsyn)
            #if any([x in phGraph1.neighbors(oldsyn) for x in otherCons]):
            #    phGraph1.remove_edge(oldsyn,currCon)
            #else:
            #    phGraph1.remove_node(oldsyn)
        else:
            print "    Dumping newsyn"
            phGraph1.remove_node(newsyn)
            #if any([x in phGraph1.neighbors(newsyn) for x in otherCons]):
            #    phGraph1.remove_edge(newsyn,currCon)
            #else:
            #    phGraph1.remove_node(newsyn)
        
        # overwrite the old network, but now wipe out activations
        phGraph0 = phGraph1.copy() 
        for currNode in phGraph0:
            phGraph0.node[currNode]['activation'] = 0.01
            
    # record the average similarity between words...by just phonemes
    # I know there is a better way to do the below, maybe with itertools,
    # but I'm just blanking on it right now
        
    currDiffList = []
    for word1 in wordList:
        for word2 in wordList:
            if word1 != word2:
                currDiffList.append( len(word1.intersection(word2)) / len(word1.difference(word2))  ) # shared over unshared is apparently how graff calculated word similarity
    diffList.append(np.mean(currDiffList))
    
    # I'll then want to either record the number of minimal pairs for each 
    # contrast, or the average similarity between words by *features*???

"""    
plt.plot(diffList)  # see how the average similarity between words changed over
                    # simulation
plt.suptitle('Average similarity does not change over generations', fontsize=30)
plt.xlabel('Generation', fontsize=24)
plt.ylabel('Average similarity between words', fontsize=24)

#plt.plot(minPairCount)
plt.show()
"""

# make sure the directory below is right....this part should probably be in its 
# own file...

os.chdir('/Users/russellrichie/Richie extension of Graff (2012)/Richie-extension-of-Graff-2012/network activation diagrams/sim1')
files = os.listdir('.')
files = [f for f in files if f[-3:] == 'png']
images = [Image.open(file) for file in files]    
images2gif.writeGif('networkAnimationDynamics.gif',images, duration=2.0, dither=0)