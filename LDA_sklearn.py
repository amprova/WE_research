import pandas as pd
import numpy as np
from scipy import sparse as sps
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import logging
from gensim import corpora
from lenskit import util
from sklearn.decomposition import LatentDirichletAllocation
logging.basicConfig(filename='LDA.log',filemode='a',level=logging.INFO)
_logger = logging.getLogger(__name__)

class LDA_SK:
    
    similarity_matrix = None
    review_data = None
    item_data = None
    timer = None
    #NUM_TOPICS = 20
    
    
    def __init__(self, tokenize = True, lower = True, stop_words_remove = True, stemmed = True, NUM_TOPICS=20):
        
        self.tokenize = tokenize
        self.lower = lower
        self.stop_words_remove = stop_words_remove
        self.stemmed = stemmed
        self.NUM_TOPICS = NUM_TOPICS
        
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
     
    def LDA(self, data_table, col_name):
        
        lda = LatentDirichletAllocation(n_components=self.NUM_TOPICS, random_state=0)
        dictvectorizer = DictVectorizer(sparse=True)
        bow = data_table[col_name].tolist()
        dictionary = corpora.Dictionary(bow) 
        corpus = [dictionary.doc2bow(text) for text in bow]
        data_table['doc2bow'] = corpus
        dict_val = data_table['doc2bow'].apply(lambda row: self.tuple_to_dict(row))
        count_vec = dictvectorizer.fit_transform(dict_val)
        
        LDA_mat = lda.fit_transform(count_vec)
        LDA_MAT = sparse.csr_matrix(LDA_mat)
        return LDA_MAT

    def inner_prod(self,mat_name):
        
        inner_prod = mat_name @ mat_name.T
        return inner_prod.toarray()

    def get_user_item(self, userID):
        user_item_ids = self.review_data.set_index('user')['item']
        user_item = user_item_ids.loc[userID]
        if isinstance(user_item, str):
            user_item = pd.Series(user_item).rename("item")
        temp_df = user_item.to_frame()
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
        
        item_rev['processed_reviews'] = item_rev['review'].apply(lambda row: self.process(row))
        self.item_data = item_rev
        
        #tf_idf_mat = self.tf_idf(self.item_data, 'processed_reviews')
        LDA_mat = self.LDA(self.item_data, 'processed_reviews')
        self.similarity_matrix = self.inner_prod(LDA_mat)
        _logger.info('[%s] fitting LDA model', self.timer)
        
        return self
    
    def predict_for_user(self, userID, itemList, ratings = None):
        user_item_ids = self.review_data.set_index('user')['item']
        if userID in user_item_ids.index:
            temp_df = self.get_user_item(userID)
            items = temp_df[:]['item']
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
        return 'LDA'
        
            
         