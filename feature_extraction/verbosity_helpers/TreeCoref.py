import pickle
import json
from stanfordcorenlp import StanfordCoreNLP
import os
import re
from nltk.tokenize import sent_tokenize
import pandas as pd

class SnippetTrees:
    def __init__(self,corenlp_path,snippet_path,snippet_loc):
        with open(snippet_path,'rb') as handle:
          dict_snippets = pickle.load(handle)
        self.snippets = dict_snippets['snippets']
        self.loc = snippet_loc
        self.nlp = StanfordCoreNLP(corenlp_path, quiet=False)
    
    def reformat_snippets(self):
        snippets_reformatted = []

        for index,snip in enumerate(self.snippets):
            snip_new = snip.replace(';','.')
            snip_new = snip_new.replace(',','.')
            snip_new = re.sub(r'[^A-Za-z.\s]', '', snip_new)
            snippet_group = sent_tokenize(snip_new)
            snippets_reformatted.append(snippet_group)
        return snippets_reformatted
        
    def generate_snippet_trees(self):
        props = {'annotators': 'parse', 'pipelineLanguage': 'en'}
        #folder path for saving tree files
        folder_path = 'trees_and_coref_links/snippet_trees/'+self.loc+'/'
        start_index = len(os.listdir(folder_path))
        snippets_reformatted = self.reformat_snippets()
        for index,i in enumerate(snippets_reformatted[start_index:]):
          print('Snippet ',start_index+index)
          trees = []
          for j in i:
            print(j)
            out = self.nlp.annotate(j, properties=props)
            try:
              output = json.loads(out)
              parse_tree = output['sentences'][0]['parse']
              trees.append(parse_tree)
            except Exception as err:
              print(err)
            finally:
              pass
          print('\n')
          with open(folder_path+str(index+start_index)+'.pickle','wb') as handle:
              pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
class SnippetCorefLinks:
    def __init__(self,corenlp_path,snippet_path,snippet_loc):
        with open(snippet_path,'rb') as handle:
          dict_snippets = pickle.load(handle)
        self.snippets = dict_snippets['snippets']
        self.loc = snippet_loc
        self.nlp = StanfordCoreNLP(corenlp_path, quiet=False)
        
    def coreference_links(self,text,prop):
 
        inter_links = 0
        intra_links = 0
        
        out = self.nlp.annotate(text, properties=prop)
        try:
          output = json.loads(out)
          corefs = []
          for i in output['corefs']:
              if len(output['corefs'][i]) > 1:
                  corefs.append(output['corefs'][i])

          sent_nums = []
          for i in corefs:
              arr_nums = [j['sentNum'] for j in i]
              sent_nums.append(arr_nums)

          for i in sent_nums:
              for j in range(len(i)):
                  if j+1 in range(len(i)):
                      if i[j] == i[j+1]:
                          intra_links = intra_links + 1
                      else:
                          inter_links = inter_links + 1
                  else:
                      break

        except Exception as err:
          print(err)
        finally:
          pass
    
        return intra_links,inter_links
    
    def generate_coref_links(self):
        props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
        folder_path = 'trees_and_coref_links/coref_links/'+self.loc+'_snippets'
        start_index = len(os.listdir(folder_path))

        for index,text in enumerate(self.snippets[start_index:]):
          dict_coref = {'intra_links':0,'inter_links':0,'coref_links':0}
          print('Snippet ',start_index+index)
          intra_links,inter_links = self.coreference_links(text,props)
          dict_coref['intra_links'] = intra_links
          dict_coref['inter_links'] = inter_links
          dict_coref['coref_links'] = inter_links+intra_links
          with open(folder_path+str(index+start_index)+'.pickle','wb') as handle:
              pickle.dump(dict_coref, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
class PatStatTrees:
    def __init__(self,epo_dir_path,corenlp_path):
        df_epo_1 = pd.read_csv(os.path.join(epo_dir_path,'1.csv'),sep=';')
        df_epo_2 = pd.read_csv(os.path.join(epo_dir_path,'2.csv'),sep=';')
        df_epo_3 = pd.read_csv(os.path.join(epo_dir_path,'3.csv'),sep=';')
        df_epo_4 = pd.read_csv(os.path.join(epo_dir_path,'4.csv'),sep=';')
        df_epo = pd.concat([df_epo_1, df_epo_2,df_epo_3,df_epo_4], ignore_index=True, sort=False)
        self.df_epo = df_epo.sample(frac=1).reset_index(drop=True)
        self.sents_epo = self.pick_random_sentences()
        self.nlp = StanfordCoreNLP(corenlp_path, quiet=False)
        
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
        props = {'annotators': 'parse', 'pipelineLanguage': 'en'}
        np_prods = []
        non_np_prods = []
        folder_path = 'trees_and_coref_links/patstat_trees'
        start_index = len(os.listdir(folder_path))
        for index,sent in enumerate(self.sents_epo[start_index:]):
            print('\n\n')
            print("Sentence ",start_index+index+1)
            print(sent)
            output = self.nlp.annotate(sent, properties=props)
            error_text = 'CoreNLP request timed out. Your document may be too long.'
            if output != error_text:
                try:
                    obj = json.loads(output)
                    parse_tree =  obj['sentences'][0]['parse']
                    with open('_trees/'+str(start_index+index)+'.pickle', 'wb') as handle:
                        pickle.dump(parse_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except JSONDecodeError:
                    print ('JSONDecodeError: ',obj)
                    
                

          
        
                                      