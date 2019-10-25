#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from scipy import sparse as sps
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import lenskit
import lenskit.crossfold as xf
from  lenskit.crossfold import TTPair


from lenskit.algorithms import als, basic, item_knn, user_knn
from lenskit.algorithms.basic import Fallback
from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms.implicit import BPR


from lenskit.batch import MultiEval
from lenskit.crossfold import partition_users, SampleN
from lenskit import batch, topn, util
from tf_idf import tf_idf

file = open("pairs_user_new.pickle","rb")
pairs_user = pickle.load(file)

truth = pd.concat((p.test for p in pairs_user))


def ndcg(file_name, truth):
    recs = pd.read_parquet(file_name)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    ndcg = rla.compute(recs, truth)
    return ndcg

def RR(file_name, truth):
    recs = pd.read_parquet(file_name)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)
    RR = rla.compute(recs, truth)
    return RR

RR_algo_comp = RR('results/steam/pruned_5_new/recommendations.parquet', truth)


legend = pd.read_csv("results/steam/pruned_5_new/runs.csv")
legend = legend.set_index('RunId').loc[:,'AlgoStr']

RR_algo = MRR_algo.join(legend, on='RunId')

pickle_out = open("results/steam/pruned_5_new/RR_algo.pickle","wb")
pickle.dump(RR_algo, pickle_out)
pickle_out.close()

