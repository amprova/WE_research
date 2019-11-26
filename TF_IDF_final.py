import pandas as pd
import numpy as np
from scipy import sparse as sps
import nltk
import pickle
import operator
from itertools import chain
import ast
import gzip
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import logging
from gensim import corpora
from lenskit import util
logging.basicConfig(filename='TF_IDF.log',filemode='a',level=logging.INFO)
_logger = logging.getLogger(__name__)
from sklearn.feature_extraction.text import CountVectorizer

class tf_idf:
    
    similarity_matrix = None
    review_data = None
    item_data = None
    timer = None
    user_index = None
    

    def __init__(self, tokenize = True, lower = True, stop_words_remove = True, stemmed = True):
        
        self.tokenize = tokenize
        self.lower = lower
        self.stop_words_remove = stop_words_remove
        self.stemmed = stemmed
        
    def process(self, content):
        
        if self.tokenize is True:
            processed = tokenizer.tokenize(content)
        else:
            processed = content.split()
        if self.lower is True:
            processed = [token.lower() for token in processed]
        if self.stop_words_remove is True:
            processed = [token for token in processed if token not in stopwords.words('english')]
        if self.stemmed is True:
            processed = [ps.stem(token) for token in processed]
        
        return processed
        
    
    def tuple_to_dict(self,row):
        dc = dict((x, y) for x, y in row)
        return dc
    
    def tf_idf(self, data_table, col_name):
        
        vect = CountVectorizer(tokenizer=self.process)
        corpus = data_table[col_name].tolist()
        BOW = vect.fit_transform(corpus)
        tfidf_transformer=TfidfTransformer()
        tf_idf_mat = tfidf_transformer.fit_transform(BOW)
        return tf_idf_mat
     
    
    def cosine_sim(self, mat_name):
        norm_mat = normalize(mat_name, norm='l2', axis=1)
        cosine_mat = norm_mat @ norm_mat.T
        return cosine_mat.toarray()

    def get_user_item(self, userID):
        #user_item_ids = self.review_data.set_index('user')['item']
        item_list = self.user_index.loc[userID]
        if isinstance(item_list, str):
            item_list = pd.Series(item_list).rename("item")
        temp_df = item_list.to_frame()
        temp_df = temp_df.reset_index()
        return temp_df
    
    def itemid_2_index(self):
        r_index = pd.Index(self.item_data.item.unique(), name='item')
        return r_index
    
    def score_reviews(self, item):
        try:
            item2index = self.itemid_2_index()
            idx = item2index.get_loc(item)
        except KeyError:
            return pd.Series(0, item2index, name='rev_sim')
        row = self.similarity_matrix[idx, :].copy()
        row[idx] = 0
        item_sim = pd.Series(row, item2index, name='rev_sim')
        return item_sim

    def fit(self, pruned_data):
       
        self.timer = util.Stopwatch()
        self.review_data = pruned_data
        only_rev = pruned_data.dropna()
        
        item_rev = pd.DataFrame({'review': only_rev.groupby(['item']).review.apply(lambda x:' '.join(x))})
        item_rev.reset_index(inplace=True)
        
        #item_rev['processed_reviews'] = item_rev['review'].apply(lambda row: self.process(row))
        self.item_data = item_rev
        
        tf_idf_mat = self.tf_idf(self.item_data, 'review')
        self.similarity_matrix = self.cosine_sim(tf_idf_mat)
        self.user_index = self.review_data.set_index('user')['item']
        _logger.info('[%s] fitting tfidf model', self.timer)
        
        return self
    
    def predict_for_user(self, userID, itemList, ratings = None):
        #user_item_ids = self.review_data.set_index('user')['item']
        if userID in self.user_index.index:
            temp_df = self.get_user_item(userID)
            items = pd.Series(temp_df['item'].unique())
            scores = items.apply(lambda x: self.score_reviews(x))

            present = items[items.isin(scores.columns)]
            scores.loc[:, present] = 0
            #final_score = scores.sum(axis=0)

            predList = scores.filter(items=itemList)
            final_score = predList.sum(axis=0)
            _logger.info('[%s] Predicting items for UserID %s', self.timer, userID)
            return final_score
        else:
            return pd.Series(np.nan, index=itemList)
        
    def __str__(self):
        return 'Tf-IDF'
    
    
        
            
         