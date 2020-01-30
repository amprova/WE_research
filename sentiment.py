import pandas as pd
import numpy as np

class sentiment_orientation:
    
    
    def __init__(self):
        return
    
    def make_merged_df(self, aspect_annot, opinion_lexicon, adj_mod):
        
        new_df = aspect_annot.drop(columns=['userID','itemID','noun'])
        new_df['adj_polarity']=""
        new_df['adj_mod']=""
        new_df['is_affirmative']=""
        
        for index in new_df.index:
            new_df['adj_polarity'][index] = self.get_padj(new_df, opinion_lexicon, index)
            new_df['adj_mod'][index] = self.get_wmod(new_df, adj_mod, index)
            new_df['is_affirmative'][index] = self.get_waff(new_df,index)
        new_df.to_parquet('new_aspect_annot_games.parquet')   
        return new_df
        
    def get_waff(self, df, index):
        
        if df['is_negated'][index]==0:
            return 1
        else:
            return 0
    
        
    def get_wmod(self, df, adj_mod, index):
        adv = df['adjective_modifier'][index]
        if pd.isnull(adv):
            return 1
        else:
            try:
                weight = adj_mod.loc[adv].weight
                return weight
            except:
                return 1 
    
    def get_padj(self, df, opinion_lexicon, index):
    
        adj = df['adjective'][index]
        try:
            polarity_list = opinion_lexicon.loc[adj]['polarity'].tolist()
            polairty = polarity_list[0]  
            return polarity
        except:
            return 0 
        
    def sentiment_calc(self, aspect_annot, opinion_lexicon, adj_mod):
        aspect_annot['SO']=''
        resulted_df = self.make_merged_df(aspect_annot, opinion_lexicon, adj_mod)
        resulted_df['SO']=resulted_df['adj_polarity']*resulted_df['adj_mod']*resulted_df['is_affirmative']
        resulted_df.to_parquet('sent_orientation_games.parquet')
        return resulted_df
 