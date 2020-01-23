#!/usr/bin/env python
# coding: utf-8

# In[107]:


import stanfordnlp
import pandas as pd
nlp = stanfordnlp.Pipeline(lang='en', treebank ='en_gum')

# 1. extractDependencies(Sentence)
# 2. getNounSiblings(d.dep)
# 3. getAdjectiveSiblings(d.gov)
# 4. isAffirmative(?)
# 5. isAdjective
# 6. isVerb
# 7. isNoun()
# 8. isPronoun()
# 9. getAdjectiveModifiers()
# 10. 

class aspect_extract:
    
    def __init__(self):
        return
        
    def extractDependencies(self, sentence):
        doc = nlp(sentence)
        dependency = doc.sentences[0].dependencies
        depTable = pd.DataFrame(dependency, columns=['gov', 'rel', 'dep'])
        return depTable

    def getNounSibling(self, depTable, nountoken, adjtoken):
        ## I add this ##
        if self.isPronoun(nountoken):
            for index in range(0,len(depTable)):
                if depTable['gov'][index].lemma == adjtoken.lemma and self.isNoun(depTable['dep'][index]):
                    noun = depTable['dep'][index].lemma
                    nounList=[noun]
        elif self.isNoun(nountoken):
            noun = nountoken.lemma
            nounList=[noun]
        for index in range(0,len(depTable)):
            if depTable['rel'][index] =='conj' and depTable['gov'][index].lemma == noun:
                nounList.append(depTable['dep'][index].lemma)
        return nounList

    def isdependent(self, depTable, newadj):
        for ind in range(0,len(depTable)):
            ## if the newadj is not a gov of nsub which is not dependant on adj
            if 'nsubj' in depTable['rel'][ind] and depTable['gov'][ind].lemma == newadj:
                return False
        return True

    def getAdjectiveSibling(self, depTable, adj, nouns):
        adjective = adj.lemma
        adjList=[adjective]
        for index in range(0,len(depTable)):
            if depTable['rel'][index] =='conj' and depTable['gov'][index].lemma == adjective:
                newadj = depTable['dep'][index].lemma
                if self.isdependent(depTable, newadj):  ## if both adj describing same aspect
                    adjList.append(newadj)
        return adjList

    def getAdjectiveModifiers(self, depTable, adj):
        #adjective = adj.lemma    
        adjmod='_'
        for index in range(0,len(depTable)):
            if depTable['rel'][index] =='advmod' and depTable['gov'][index].lemma == adj:
                adjmod = depTable['dep'][index].lemma
        return adjmod

    def isNoun(self, token):
        POS = token.xpos
        if 'NN' in POS:
            return True
        return False

    def isPronoun(self, token):
        POS = token.xpos
        if 'PR' in POS:
            return True
        return False

    def isAdjective(self, token):
        POS = token.xpos
        if 'JJ' in POS:
            return True
        return False

    def isVerb(self, token):
        POS = token.xpos
        if 'VB' in POS:
            return True
        return False

    def isAdverb(self, token):
        POS = token.xpos
        if 'RB' in POS:
            return True
        return False

    def makeNeg(self, depTable, Col):
        for index in range(0,len(depTable)):
            if depTable['rel'][index] =='advmod' and 'Neg' in depTable[Col][index].feats:
                depTable['rel'][index] ='neg'
        return depTable


    def isAffirmative(self, depTable, adj):
        #adjective = adj.lemma
        for index in range(0,len(depTable)):
            if depTable['rel'][index] =='neg' and depTable['gov'][index].lemma == adj:
                return False
        return True


    def annotations(self, nouns, adj, mod, aff):
        #a =[]
        for n in nouns:
            t = tuple((n, adj, mod, aff))
        return t

    def casensub(self, df, index):
        dep = df['dep'][index]
        gov = df['gov'][index]           # getting the dependence of the noun (looking for adj or vb)
        nouns = self.getNounSibling(df, dep, gov)  #getting the nouns that are joined by conjuction
        a =[]
        if self.isAdjective(gov):            ## example: the room is big
            adjs = self.getAdjectiveSibling(df, gov, nouns)  #getting the adjective that are joined by conj
            for adj in adjs:                     #for all the adj check the modifier and affirmative
                mod = self.getAdjectiveModifiers(df, adj)
                aff = self.isAffirmative(df, adj)
                a.append(self.annotations(nouns, adj, mod, aff))
            return a
        elif self.isVerb(gov):         #example:The staff works fast
            for ind in range(0, len(df)):
                if (df['rel'][ind] == 'xcomp' or df['rel'][ind] == 'advmod') and df['gov'][ind].lemma == gov.lemma:    ##looking for the adjective or adverb
                    newdep = df['dep'][ind]                ## adj or adv
                    adjs = self.getAdjectiveSibling(df, newdep, nouns)   ## list of adj joined by conj
                    for adj in adjs:                     #for all the adj check the modifier and affirmative
                        mod = self.getAdjectiveModifiers(df, adj)
                        aff = self.isAffirmative(df, adj)
                        a.append(self.annotations(nouns, adj, mod, aff))
                    return a

    def caseamod(self, df, index):   # example: The restaurant has good staff
        dep = df['dep'][index]
        gov = df['gov'][index]
        a =[]
        if self.isAdjective(dep) or self.isAdverb(dep) and (self.isNoun(gov) or self.isPronoun(gov)):
            nouns = self.getNounSibling(df, gov, dep)  #gov is a noun and dep is an adj
            adjs = self.getAdjectiveSibling(df, dep, nouns)  # dep is the adj
            for adj in adjs:                     #for all the adj check the modifier and affirmative
                mod = self.getAdjectiveModifiers(df, adj)
                aff = self.isAffirmative(df, adj)
                a.append(self.annotations(nouns, adj, mod, aff))
            return a


    def getsubj(self, depTable, verb):
        for ind in range(0,len(depTable)):
            if df['rel'][ind] == 'nsubj' and df['gov'][ind].lemma == verb.lemma:
                #print(df['dep'][index])
                return df['dep'][ind]


    def casecomp(self, df, index):
        dep = df['dep'][index]  #adj
        gov = df['gov'][index]  #verb
        a =[]
        if self.isAdjective(dep):
            sub = self.getsubj(df, gov)
            nouns = self.getNounSibling(df, sub, dep)  #(df, n, adj)
            adjs = self.getAdjectiveSibling(df, dep, nouns)
            for adj in adjs:                     #for all the adj check the modifier and affirmative
                mod = self.getAdjectiveModifiers(df, adj)
                aff = self.isAffirmative(df, adj)
                a.append(self.annotations(nouns, adj, mod, aff))

            return a

    def aspect_extraction(self, text):

        df = self.extractDependencies(text)
        df = self.makeNeg(df, 'dep')

        A = []
        for index in range(0, len(df)):
            if 'nsubj' in df['rel'][index] and self.isNoun(df['dep'][index]):
                annot = self.casensub(df, index)
                if annot is not None and annot not in A:
                    A.append(annot)
            elif 'amod' in df['rel'][index]:
                annot = self.caseamod(df, index)
                if annot is not None and annot not in A:
                    A.append(annot)
            elif 'comp' in df['rel'][index] and self.isVerb(df['gov'][index]):
                annot = self.casecomp(df, index)
                if annot is not None and annot not in A:
                    A.append(annot)
        return A

