import sys
sys.path.append('/Users/deepak/Desktop/thesis_pqai/')
sys.path.append('/Users/deepak/Desktop/thesis_pqai/specificity_model/')

from verbosity_helpers.Description import DescProperties
from specificity_model.helpers.ExtractFeatures import ExtractFeatures,ReadFeatureFiles
import pandas as pd
import numpy as np
import os
import pickle

class VerbosityFeatures:
    def __init__(self,path_to_big_patent,snippet_location):       #could be start or end
        df = pd.read_csv(path_to_big_patent)
        self.desc_prop = DescProperties(df.abstract.values,snippet_location)
        self.desc_prop.prepare_snippets()
        self.snippet_location = snippet_location
        
        
    def extract_features(self):
        epo_dir_path = 'data/patstat'
        df_epo_1 = pd.read_csv(os.path.join(epo_dir_path,'1.csv'),sep=';')
        df_epo_2 = pd.read_csv(os.path.join(epo_dir_path,'2.csv'),sep=';')
        df_epo = pd.concat([df_epo_1, df_epo_2], ignore_index=True, sort=False)
        df_epo = df_epo.sample(frac=1).reset_index(drop=True)
        df_epo = df_epo.rename({'appln_abstract': 'abstract'}, axis=1)
        
        #1. length_of_units

        dict_len_units = self.desc_prop.length_of_units()
        self.store_features_as_file('len_units.pickle',dict_len_units)
        
        dict_discourse_relations = self.desc_prop.discourse_relations_features()
        self.store_features_as_file('discourse_relations.pickle',dict_discourse_relations)
        
        dict_continuity = self.desc_prop.continuity()
        self.store_features_as_file('continuity.pickle',dict_continuity)
        
        dict_amount_of_detail = self.desc_prop.amount_of_detail(df_epo) #epo data dataframe
        self.store_features_as_file('amount_of_detail.pickle',dict_amount_of_detail)
        
        
        dict_syntactic_realization = self.desc_prop.syntactic_realization()
        self.store_features_as_file('syntactic_real.pickle',dict_syntactic_realization)
        
        dict_compression_likelihood = self.desc_prop.compression_likelihood()
        self.store_features_as_file('compression_likelihood.pickle',dict_compression_likelihood)
        
        
    def store_features_as_file(self,file_name,feature_dict):
        folder_path = os.path.join('verbosity_feats',self.snippet_location)
        file_path = os.path.join(folder_path,file_name)
        with open(file_path, 'wb') as handle:
                  pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(file_name,' stored in ',folder_path,' folder.')
        
    def read_feature_files(self,loc):
        folder_path = os.path.join('verbosity_feats',self.loc)
        return folder_path
        