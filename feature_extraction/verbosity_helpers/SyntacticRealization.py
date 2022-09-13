import pandas as pd
from nltk.tokenize import sent_tokenize
import pickle
import os
#from pycorenlp import StanfordCoreNLP
import json
from nltk import Tree
from collections import Counter
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
import os

import re

class TreeProductions:
    def __init__(self,epo_dir_path):        
        df_epo_1 = pd.read_csv(os.path.join(epo_dir_path,'1.csv'),sep=';')
        df_epo_2 = pd.read_csv(os.path.join(epo_dir_path,'2.csv'),sep=';')
        df_epo_3 = pd.read_csv(os.path.join(epo_dir_path,'3.csv'),sep=';')
        df_epo_4 = pd.read_csv(os.path.join(epo_dir_path,'4.csv'),sep=';')
        df_epo = pd.concat([df_epo_1, df_epo_2,df_epo_3,df_epo_4], ignore_index=True, sort=False)
        self.df_epo = df_epo.sample(frac=1).reset_index(drop=True)
        self.sents_epo = self.pick_random_sentences()
        print('Please run CoreNLP server if are generating parse trees.')
        
    def trees_available(self):
        self.tree_count = len(os.listdir('trees_and_coref_links/patstat_trees'))
        if self.tree_count == 0:
          return False
        else:
          return True
                      
        
    def pick_random_sentences(self):
        file_path = 'sents_epo.pickle'
        if os.path.exists(file_path):
            with open('sents_epo.pickle', 'rb') as handle:
                sents_epo = pickle.load(handle)
        else:
            sents_epo = []
            for abstract in self.df_epo.appln_abstract.values:
                if len(sents_epo) < 50000:
                    abstract = abstract.replace(';','.')
                    abstract = abstract.replace(',','.')
                    sents = sent_tokenize(abstract)
                    sents_epo = sents_epo + sents
                else:
                    break
            with open(file_path, 'wb') as handle:
                pickle.dump(sents_epo, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return sents_epo
    
    def generate_trees(self):
        if self.trees_available() and self.tree_count >= 50000:
            print("Tree are available in 'new_trees' folder. Read them using read_productions_from_tree_files() method.")
        else:
            if self.tree_count == 0:
                start_index = 0
            else:
                print(self.tree_count-1,' trees are already generated. Starting from where you left.')
                start_index = self.tree_count - 1
            
            nlp = StanfordCoreNLP('http://localhost:9000')
            np_prods = []
            non_np_prods = []

            #start_index = 0
            for index,sent in enumerate(self.sents_epo[start_index:]):
                print('\n\n')
                print("Sentence ",start_index+index+1)
                print(sent)
                output = nlp.annotate(sent, properties={'annotators': 'parse','outputFormat': 'json','timeout': 400000})
                error_text = 'CoreNLP request timed out. Your document may be too long.'
                if output != error_text:
                    try:
                        obj = json.loads(output)
                        parse_tree =  obj['sentences'][0]['parse']
                        with open('new_trees/'+str(start_index+index)+'.pickle', 'wb') as handle:
                            pickle.dump(parse_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except JSONDecodeError:
                        print ('JSONDecodeError: ',obj)
                        
    def generate_snippet_trees(self,snippets,snippet_loc):     #snippet_loc = {start, end}
        snippets_reformatted = []
        for index,snip in enumerate(snippets):
            snip_new = snip.replace(';','.')
            snip_new = snip_new.replace(',','.')
            snip_new = re.sub(r'[^A-Za-z.\s]', '', snip_new)
            snippet_group = sent_tokenize(snip_new)
            snippets_reformatted.append(snippet_group)
            
        
        nlp = StanfordCoreNLP(r'/Users/deepak/Downloads/stanford-corenlp-4.4.0', quiet=False)
        props = {'annotators': 'parse', 'pipelineLanguage': 'en'}
        
        #nlp = StanfordCoreNLP('http://localhost:9000')
        #start_index = 0
        for index,sent_group in enumerate(snippets_reformatted[start_index:]):
            tree_group = []
            print('\n')
            print('Sentence ',start_index+index,':')
            for sent in sent_group:
                print(sent)
                output = nlp.annotate(sent, properties=props)
                obj = json.loads(output)
                parse_tree =  obj['sentences'][0]['parse']
                tree_group.append(parse_tree)
            with open('trees_and_coref_links/snippet_trees/'+self.snippet_loc+'/'+str(index)+'.pickle', 'wb') as handle:
                pickle.dump(tree_group, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
    def read_productions_from_tree_files(self):
        if self.trees_available():
            tree_files = os.listdir('trees_and_coref_links/patstat_trees')
            dict_productions = {'with_nps':[],'wo_nps':[]}
            for index,tree_file in enumerate(tqdm(tree_files,desc='Collecting productions from parse trees.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
                with open('new_trees/'+tree_file, 'rb') as handle:
                    tree = pickle.load(handle)
                t = Tree.fromstring(tree)
                nps = []
                non_nps = []
                for prod in t.productions():
                    if prod.lhs().symbol() == 'NP':
                        nps.append(prod)
                    else:
                        non_nps.append(prod)
                dict_productions['with_nps'] = dict_productions['with_nps'] + nps
                dict_productions['wo_nps'] = dict_productions['wo_nps'] + non_nps

            self.dict_productions = dict_productions
            return self.dict_productions
        else:
            print('Tree folder is empty.')
            return 'Tree folder is empty.'
    
    def most_common_productions(self,count):
        nps_counts = Counter(self.dict_productions['with_nps'])
        non_nps_counts = Counter(self.dict_productions['wo_nps'])
        top_nps = nps_counts.most_common(count)
        top_non_nps = non_nps_counts.most_common(count)
        self.dict_frequent_prods = {'freq_nps':top_nps,'freq_non_nps':top_non_nps}
        with open('frequent_productions.pickle', 'wb') as handle:
            pickle.dump(self.dict_frequent_prods, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.dict_frequent_prods