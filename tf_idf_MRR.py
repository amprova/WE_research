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

from fastparquet import ParquetFile

result = pd.DataFrame()
pf = ParquetFile('results/steam/pruned_5_new/recommendations.parquet')
for df in pf.iter_row_groups():
    trancate = df.loc[df['rank']<1001]
    result = result.append(trancate,sort = False)

#result.to_parquet('results/steam/pruned_5_new/recs.parquet')
#result.to_csv("results/steam/pruned_5_new/recs.csv")


def RR(rec, truth):
    #recs = pd.read_parquet(file_name)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.recip_rank)
    RR_result = rla.compute(rec, truth)
    return RR_result

RR_algo_comp = RR(result, truth)
print("done with recip_rank")

RR_algo_comp.to_parquet('results/steam/pruned_5_new/RR_algo1000.parquet')
RR_algo_comp.to_csv("results/steam/pruned_5_new/RR_algo1000.csv", ignore_index=True)

pickle_out = open("results/steam/pruned_5_new/RR_algo1000.pickle","wb")
pickle.dump(RR_algo_comp, pickle_out, protocol = 4)
pickle_out.close()


### join with algo name
legend = pd.read_csv("results/steam/pruned_5_new/runs.csv")
legend = legend.set_index('RunId').loc[:,'AlgoStr']

RR_algo = RR_algo_comp.join(legend, on='RunId')

RR_algo.to_parquet('results/steam/pruned_5_new/RR_algoname1000.parquet')
RR_algo.to_csv("results/steam/pruned_5_new/RR_algoname1000.csv", ignore_index=True)

pickle_out = open("results/steam/pruned_5_new/RR_algoname1000.pickle","wb")
pickle.dump(RR_algo, pickle_out, protocol = 4)
pickle_out.close()

print("done with merging")

## saving result
pickle_out = open("results/steam/pruned_5_new/recs1000.pickle","wb")
pickle.dump(result, pickle_out, protocol=4)
pickle_out.close()
#result.to_parquet('results/steam/pruned_5_new/recs.parquet')
#result.to_csv("results/steam/pruned_5_new/recs.csv")







