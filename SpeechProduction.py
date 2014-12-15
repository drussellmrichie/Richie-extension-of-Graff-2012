# -*- coding: utf-8 -*-

from __future__ import division

"""
REIMPLEMENTATION OF MARTIN (2007) / GRAFF (2012) SPREADING ACTIVATION MODEL OF
SPEECH PRODUCTION

Do I need to let activation feed back up to concept layers? I don't think Martin did. For now, I am.

I should run this in pycharm in debug mode so I can see what's what....

What should (negative) weights between words be? Right now, I just set to -1....? Definitely need to think about how this will work in computation
of net inputs and new activations, etc. I think it needs to be higher to offset the positive weighted inputs from the phoneme layer...right now
there is runaway activation....and with greater numbers of concepts and consequently words, the target words never win out in the end!!! And actually,
the activation appears to even out across all the words as the settling time increases. ALSO ALSO, one synonym doesn't seem to ever 'beat' the 
other....

Consider restricting set of phonemes and features to make the word space smaller and hence minimal pairs more likely....
    Think about the minimum number of concepts/words, phonemes, and features to really show the phenomenon....also, the fewer we have, the easier
    it'd be to manage the TRACE temporal fudge....

Martin footnote 19 Only weights for lexical nodes are given random weights. All other nodes (those for concepts and phonemes) have weights of 1.0.

Also need to figure out exactly how Martin and Graff incremented resting
activation as a function of phoneme frequency? Is it the same as word-node 
weight update in Martin on paper page 22? Do I have to update weights or resting
activation? ACTUALLY, I'M NOT SURE THAT MARTIN DID THIS????

    weights for each node, which is intended to simulate differing resting activations
    
    (9) Speaker weight adjustment
    WS ←WS +α(e−WS ) (10) Listener weight adjustment
    WL ←WL +β(e−WL )

    I might not need to do this, since Martin and Graff had weights/resting act
    change to model frequency distributions across phonemes?

Things to consider...how many words? how many characters per word? should I 
restrict the phoneme set and feature set?

I think I didn't think through the serial order aspect of words. In Martin and
Graff's models, words were really conceived of as sets of simultaneous phonemes.
But that doesn't work for us. We can't have the different phonemes simultaneously
activating their corresponding features. Can I get away with having single-phoneme
words?

Would we have to do some kind of TRACE fudge where we make a different phoneme
for each slot, i.e., an esh1, esh2, esh3, etc.? Or, is there some way to activate
the phonemes sequentially (step through them)? Let activation flow from concept
down to features and back up to lexical level, then activate next phoneme?

^^^This still matters....11/24/2014....but is it so different than martin/graff's strategy of dealing with words with more than one 'a'?

Might consider taking the triangle model and Dell type solution and put one set
of phonemes in onset, another in nucleus, and another in coda. And have separate
set of features for onset, nucleus, and coda.

Getting double linkage between words because of the way the loop is built.
"""

from numpy import genfromtxt
import numpy as np
import numpy.random
import random, os #, itertools
import networkx as nx
import matplotlib.pyplot as plt
#import pygraphviz as pgv # pygraphviz should be available

os.chdir('/Users/russellrichie/Google Drive/UConn/Classes/DLC (II)')

# Set some parameters

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

nGens = 1       # the number of generations to be simulated

initAct = 100   # the initial activation of the concept to be expressed

eWeights = .3   # I need to read Martin more closely to see what exactly he set 
                # his weights to. He seems to set them to different values for 
                # dif sims
                
nWeights = 1    # eWeights is edge weights, which in footnote 11 Martin says he 
                # set to .3 for all connections. nWeight is node weight, which 
                # I am leaving at 1, also consistent with Martin on page 35
decay = 0.6     # decay rate

settleTime = 100  # how many 'round-trips' in the network does this allow? Martin allowed
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
        newWords.append(frozenset(random.sample(phonemes, wordLengths))) # must be frozen (hashable) so networkx can make nodes out of words
    #for _ in range(wordCount):
    #    newWords.append([random.choice(phonemes) for _ in range(wordLengths)])
    return newWords

def coinOneSyn(phonemes,wordLengths):
    return frozenset(random.sample(phonemes, wordLengths))

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
            
            """
            # Some code to try to plot the network and its nodes' activations at each time step
            # I can't get it to work because I'm having import problems! My Canopy python doesn't 
            # recognize pygraphviz and my default python in terminal doesn't recognize 
            # matplotlib.pyplot !!!! Would really love to get this sorted.
            
            A = nx.to_agraph(phGraph0)
            three = A.add_subgraph(range(conceptCount),rank='same')
            two = A.add_subgraph(wordList,rank='same')
            one = A.add_subgraph(phonemes,rank='same')
            zero = A.add_subgraph(list(distFeatData[0][1:]),rank='same')
            
            activations = [0] * len(phGraph0)
            for index, currNode in enumerate(phGraph0):
                activations[index] = phGraph0.node[currNode]['activation']

            pos=nx.graphviz_layout(phGraph0,prog='dot')

            nx.draw_networkx_nodes(A,
                       pos,
                       node_color= activations, # not sure if I have to convert activations to points on some color scale
                       node_size=500,
                        alpha=0.8)
                        
            filename = 'network architecture time{}.png'.format(currStep)
            A.draw(filename, prog='dot')
            """
            
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