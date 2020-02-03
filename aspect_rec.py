import pandas as pd
import numpy as np
import scipy
import nltk
import pickle
import operator
from itertools import chain
import ast
import gzip
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
from scipy import sparse
from sklearn.preprocessing import normalize
import logging
from gensim import corpora
from lenskit import util

logging.basicConfig(filename='aspect.log',filemode='a',level=logging.INFO)
_logger = logging.getLogger(__name__)

class aspect_recom:
    
    timer = None
    user_data=pd.DataFrame()
    item_data=pd.DataFrame()
    #NUM_TOPICS = 20
    
    
    def __init__(self):
        return

    def get_userfeature(self, df):
        user_feat=pd.DataFrame()
        user_feat['avg_SO'] =  df.groupby(['user','aspect'])['SO'].mean()
        user_feat.reset_index(inplace=True)
        user_prof = user_feat.pivot(index='user',columns='aspect',values='avg_SO')
        user_prof=user_prof.fillna(-1000)
        user_profile = scipy.sparse.csr_matrix(user_prof.values)
        return user_profile
    
    def get_itemfeature(self, df):
        item_feat=pd.DataFrame()
        item_feat['avg_SO'] =  df.groupby(['item','aspect'])['SO'].mean()
        item_feat.reset_index(inplace=True)
        item_prof = item_feat.pivot(index='item',columns='aspect',values='avg_SO')
        item_prof=item_prof.fillna(-1000)
        item_profile = scipy.sparse.csr_matrix(item_prof.values)
        return item_profile
    
    def cosine_sim(self, user_prof, item_prof):
        norm_user = normalize(user_prof, norm='l2', axis=1)
        norm_item = normalize(item_prof, norm='l2', axis=1)
        cosine_mat = norm_user @ norm_item.T
        return cosine_mat.toarray()
    
    def get_user_item(self, userID):
        #user_item_ids = self.review_data.set_index('user')['item']
        item_list = self.user_index.loc[userID]
        if isinstance(item_list, str):
            item_list = pd.Series(item_list).rename("item")
        temp_df = item_list.to_frame()
        temp_df = temp_df.reset_index()
        return temp_df
    
    
    def user_2_index(self):
        
        u_index = pd.Index(self.user_data.user.unique(), name='user')
        return u_index
        
    
    def itemid_2_index(self):
        r_index = pd.Index(self.item_data.item.unique(), name='item')
        return r_index
    
    def score_reviews(self, user):
        
        try:
            user2index = self.user_2_index()
            udx = user2index.get_loc(user)
        except KeyError:
            return pd.Series(0, itemid_2_index, name='item_sim')
        row = self.similarity_matrix[udx, :].copy()
        item_sim = pd.Series(row, self.itemid_2_index(), name='item_sim')
        item_sim_df = item_sim.to_frame()
        return item_sim_df.T


    
    def fit(self, pruned_data):
        
        self.timer = util.Stopwatch()
        self.user_index = pruned_data.set_index('user')['item']
        self.user_data['count'] = pruned_data.groupby('user')['SO'].count()
        self.user_data.reset_index(inplace=True)
        self.item_data['count'] = pruned_data.groupby('item')['SO'].count()
        self.item_data.reset_index(inplace=True)
        user_prof = self.get_userfeature(pruned_data)
        item_prof = self.get_itemfeature(pruned_data)
        self.similarity_matrix = self.cosine_sim(user_prof, item_prof)
        #_logger.info('[%s] fitting LDA model', self.timer)
        
        return self
    
    
    def predict_for_user(self, userID, itemList, ratings = None):
        #user_item_ids = self.review_data.set_index('user')['item']
        if userID in self.user_index.index:
            
            temp_df = self.get_user_item(userID)  ## df of user owned item
            user_items = pd.Series(temp_df['item'].unique())  #list of user_owned items
            scores = self.score_reviews(userID)
            
            present = user_items[user_items.isin(scores.columns)]
            scores.loc[:, present] = 0
            print(scores)
            predList = scores.filter(items=itemList)
            print(predList)
            #final_score = predList.sum(axis=0)
            _logger.info('[%s] Predicting items for UserID %s', self.timer, userID)
            return predList.T
        else:
            return pd.Series(np.nan, index=itemList)
           
        
    def __str__(self):
        return 'aspect'
        
            
         