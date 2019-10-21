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


class tf_idf:
    
    similarity_matrix = None
    review_data = None
    item_data = None
    
    #tokenize = True
    #lower = True
    #stop_words_remove = True
    #stemmed = True
    
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
     
    
    def tf_idf(self, data_table, col_name):
        corpus_list = []
        for item in data_table[col_name]:
            corpus_list.append(' '.join(item))
            #corpus_list.append(item)
        tfidf_matrix = TfidfVectorizer().fit_transform(corpus_list)
        return tfidf_matrix
    
    def cosine_sim(self, mat_name):
        norm_mat = normalize(mat_name, norm='l2', axis=1)
        cosine_mat = norm_mat @ norm_mat.T
        return cosine_mat.toarray()

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

    def get_popular_item(self):
        game_count = pd.DataFrame()
        game_count['count'] = self.review_data.groupby('item')['user'].count()
        pop = game_count.sort_values(by=['count'], ascending = False)
        popular_item = pop.head(5)
        popular_item.reset_index(inplace= True)
        return popular_item['item']

    
    def fit(self, review_data):
        
        self.review_data = review_data
        #print(type(review_data['review']))
        item_data1 = pd.DataFrame({'review': self.review_data.groupby(['item']).review.apply(lambda x:' '.join(x))})
        #print(type(item_data1))
        item_data1.reset_index(inplace=True)
        #print(item_data1)
        item_data1['processed_reviews'] = item_data1['review'].apply(lambda row: self.process(row))
        self.item_data = item_data1
        
        tf_idf_mat = self.tf_idf(self.item_data, 'processed_reviews')
        self.similarity_matrix = self.cosine_sim(tf_idf_mat)
        
        return self
    
    def predict_for_user(self, userID, itemList, ratings = None):
        #popular_items = self.get_popular_item()
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

                #final_score = score[itemID].sum()

            return final_score
        else:
            return pd.Series(np.nan, index=items)
        
    def __str__(self):
        return 'Tf-IDF'
        
            
         