from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import re
import nltk
import spacy
import os
from nltk.parse import stanford
import json
from pycorenlp import StanfordCoreNLP
from jnius import autoclass
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import groupby
from verbosity_helpers.SyntacticRealization import TreeProductions
import pickle
from tqdm import tqdm
from nltk import Tree
import sys
sys.path.append('/Users/deepak/Desktop/thesis_pqai/')
sys.path.append('/Users/deepak/Desktop/thesis_pqai/specificity_model/')
from specificity_model.helpers.ExtractFeatures import ExtractFeatures,ReadFeatureFiles
from verbosity_helpers.SentenceCompression import CompreesionLikelihood

class DescProperties:
    def __init__(self, descriptions,snippet_selection):    #Snippet selection values: {'start','end'}
        self.descriptions = descriptions
        self.discourse_relations = {'temporal':0, 'comparison':0, 'contingency':0,'expansion':0}
        self.snippet_selection = snippet_selection

        model = 'stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz'
        jar = 'stanford-ner-2020-11-17/stanford-ner.jar'
        self.ner_tagger = StanfordNERTagger(model, jar,encoding='utf-8')
        
        self.nlp = spacy.load('en_core_web_sm')
        
        #Tree production object
        file_path = 'frequent_productions.pickle'
        if os.path.exists(file_path):
            print('Most frequent grammar productions are available.')
            with open(file_path, 'rb') as handle:
                self.most_common_productions = pickle.load(handle)
                
        else:
            self.obj_tp = TreeProductions('data/patstat')
            prod = self.obj_tp.read_productions_from_tree_files()
            error_msg = 'Tree folder is empty.'
            if prod == error_msg:
                print(error_msg,'\nGenerate them using generate_trees() method from verbosity_helpers.SyntacticRealization -> TreeProductions class.')
               
            else:
                self.most_common_productions = self.obj_tp.most_common_productions(count=15)
          
             
    def count_words(self,sentence):
        return len(re.findall(r'\w+', sentence))

    def count_characters_in_word(self,word):
        return len([ele for ele in word if ele.isalpha()])

    def get_phrases(self,doc,pos_type):
        "Function to get PPs from a parsed document."
        phs = []
        for token in doc:
            # Try this with other parts of speech for different subtrees.
            if token.pos_ == pos_type:
                pp = ' '.join([tok.orth_ for tok in token.subtree])
                phs.append(pp)
        return phs
  
    def tag_ner(self,text):
        tokenized_text = nltk.word_tokenize(text)
        classified_text = self.ner_tagger.tag(tokenized_text)
        entities = []

        for tag, chunk in groupby(classified_text, lambda x:x[1]):
            if tag != "O":
                entities.append(' '.join(w for w, t in chunk))
        entities_unique = list(set(entities)) #unique entities   
        return entities_unique
    
    def coreference_links(self,text):
        nlp = StanfordCoreNLP(r'http://localhost:9000')
        props = {'annotators':'dcoref','outputFormat':'json','ner.useSUTime':'false','timeout': 800000}
        output = nlp.annotate(text, properties=props)
        corefs = []
        try:
            obj_out = json.loads(output)['corefs']
            for i in obj_out:
                if len(obj_out[i]) > 1:
                    corefs.append(obj_out[i])
         
            sent_nums = []
            for i in corefs:
                arr_nums = [j['sentNum'] for j in i]
                sent_nums.append(arr_nums)
        except Exception as err:
            print(err)    
            
        inter_links = 0
        intra_links = 0

        for i in sent_nums:
            for j in range(len(i)):
                if j+1 in range(len(i)):
                    if i[j] == i[j+1]:
                        intra_links = intra_links + 1
                    else:
                        inter_links = inter_links + 1
                else:
                    break
        return intra_links,inter_links
    
    def predict_specificity(self):
        if self.snippet_selection == 'start':
            feat_folder_path = 'features/start_snippet_feats'
        elif self.snippet_selection == 'end':
            feat_folder_path = 'features/end_snippet_feats'
            
        feat_files = ['necd_features.pickle','polarity_features.pickle','sentence_length_features.pickle',
                      'specificity_features.pickle','syntactic_features.pickle','lm_features.pickle']
        
        feats = {}
        print('Collecting specificity features.')
        for i in feat_files:
            with open(os.path.join(feat_folder_path,i),'rb') as handle:
                dict_f = pickle.load(handle)
            feats.update(dict_f)
            
        #word feats
        with open(os.path.join(feat_folder_path,'word_features.pickle'),'rb') as handle:
            dict_f = pickle.load(handle)
            print()
            word_feats = dict_f['word_feat']
       
        df_feats = pd.DataFrame(feats)
        X = np.concatenate((df_feats.values,word_feats.todense()),axis = 1)
                
        with open('/Users/deepak/Desktop/thesis_pqai/specificity_model/models/instantiation/specificity_model.pickle',
                      'rb') as f:
            clf = pickle.load(f)
        print('Predicting sentence specificity.')
        y_pred = clf.predict_proba(np.asarray(X))
        y_pred_int = np.argmax(1*(y_pred > 0.5),axis=1)
                 
        
        all_snip_with_ids = []
        for index,group in enumerate(self.dict_snippets['snippet_groups']):
            list_el = []
            for sent in group:
                list_el.append([index,sent])
            all_snip_with_ids = all_snip_with_ids + list_el
        df_temp = pd.DataFrame(all_snip_with_ids,columns=['snippet_id','text'])
        print(df_temp.shape)
        print(y_pred_int)
        df_temp['pred_label'] = y_pred_int
        print(df_temp.head(10))
        
        return feats,df_temp
    
    def prepare_snippets(self):
        k = 30
        dict_snippets = {'snippets':[],'snippet_groups':[]}
        all_selected_abstracts = []
        abstract_len = []
        for text in self.descriptions:
            text_len = self.count_words(text)
            if text_len > k:
                abstract_len.append(text_len)
                text = text.replace(';','.')
                text = text.replace(',','.')
                sentences = sent_tokenize(text)

                snippet_group = []
                last_count = 0
                if self.snippet_selection == 'start':
                    for sent in sentences:
                      if last_count < k:
                        snippet_group.append(sent.strip())
                        last_count = last_count + self.count_words(sent)   
                      else:
                        break
                    dict_snippets['snippets'].append(' '.join(snippet_group))
                    all_selected_abstracts.append(text)

                elif self.snippet_selection == 'end': 
                    end_sentences = list(reversed(sentences))            
                    for sent in end_sentences:
                      if last_count < k:
                        snippet_group.append(sent.strip())
                        last_count = last_count + self.count_words(sent)   
                      else:
                        break
                    dict_snippets['snippets'].append('. '.join(list(reversed(snippet_group))))
                    all_selected_abstracts.append(text)
                dict_snippets['snippet_groups'].append(snippet_group)
        self.dict_snippets = dict_snippets
        with open('snippets.pickle', 'wb') as handle:
                  pickle.dump(dict_snippets, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        with open('all_articles.pickle', 'wb') as handle:
                  pickle.dump(all_selected_abstracts, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        with open('labels/abstract_len.pickle', 'wb') as handle:
            pickle.dump(abstract_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return dict_snippets
        
    def length_of_units(self):
        dict_length_of_units = {'num_sents':[],'avg_sent_len':[],'avg_word_length':[],'ttr':[],
                                'np_counts':[],'avg_np_lengths':[],'vp_counts':[],
                                'avg_vp_lengths':[],'pp_counts':[],'avg_pp_lengths':[]}
        
        nlp = spacy.load('en_core_web_sm')
        snippets = self.dict_snippets['snippets']
        for index,text in enumerate(tqdm(snippets,desc='Collecting length of unit features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
                       
            # 1. number of sentences & 2. average sentence length in words
            snippet_group = self.dict_snippets['snippet_groups'][index]
            sent_count = len(snippet_group)
            count_sum = 0
            for sent in snippet_group:
              count_sum = count_sum + self.count_words(sent)
            avg_len = count_sum/sent_count
            dict_length_of_units['avg_sent_len'].append(avg_len)
            dict_length_of_units['num_sents'].append(sent_count)

            # 3. average word length in the snippet
            word_character_counts = [self.count_characters_in_word(word) for word in self.dict_snippets['snippets'][index]]
            avg_word_length = sum(word_character_counts)/len(word_character_counts)
            dict_length_of_units['avg_word_length'].append(avg_word_length)
            
            
            # 4. type to token ratio
            document = re.sub(r'[^\w]', ' ', self.dict_snippets['snippets'][index])
            document = document.lower()
            tokens = nltk.word_tokenize(document)
            types = nltk.Counter(tokens)
            ttr = (len(types)/len(tokens))*100
            dict_length_of_units['ttr'].append(ttr)


            # 5,6 noun phrases
            doc1 = nlp(text)
            noun_phrases = list(doc1.noun_chunks)
            np_count = len(noun_phrases)
            dict_length_of_units['np_counts'].append(np_count)
            
            np_lengths = [self.count_words(i.text) for i in noun_phrases]
            avg_np_length = 0
            if len(np_lengths) != 0:
                avg_np_length = sum(np_lengths)/len(np_lengths)
            dict_length_of_units['avg_np_lengths'].append(avg_np_length)
            

            # 7,8 verb phrases
            verb_phrases = self.get_phrases(doc1,pos_type= 'VERB')
            vp_count = len(verb_phrases)
            dict_length_of_units['vp_counts'].append(vp_count)
            vp_lengths = [self.count_words(i) for i in verb_phrases]
            avg_vp_length = 0
            if len(vp_lengths) != 0:
                avg_vp_length = sum(vp_lengths)/len(vp_lengths)
            dict_length_of_units['avg_vp_lengths'].append(avg_vp_length)

            
            # 9,10 Preposition phrases
            prep_phrases = self.get_phrases(doc1,pos_type = 'ADP')
            pp_count = len(prep_phrases)#
            dict_length_of_units['pp_counts'].append(pp_count)
            pp_lengths = [self.count_words(i) for i in prep_phrases]
            avg_pp_length = 0
            if len(pp_lengths) != 0:
                avg_pp_length = sum(pp_lengths)/len(pp_lengths)
            dict_length_of_units['avg_pp_lengths'].append(avg_pp_length) 
        self.dict_length_of_units = dict_length_of_units
        return dict_length_of_units
    
    def discourse_relations_features(self):
        # Run StanfortCoreNLP server before using this method.
        #nlp = StanfordCoreNLP('http://localhost:9000')
        print('in function')
        AddDiscourse = autoclass('AddDiscourse')
        
        getInstance = getattr(AddDiscourse,'getInstance')
        identifier = AddDiscourse.getInstance()
        analyze = getattr(AddDiscourse,'analyze')
        
        dict_discourse_rel = {'temporal':[], 'comparison':[], 'contingency':[],'expansion':[]}
        
        snippets = self.dict_snippets['snippets']
        for index,text in enumerate(tqdm(snippets,desc='Collecting discourse relation features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            
            discourse_relations = {'temporal':0, 'comparison':0, 'contingency':0,'expansion':0}
                        
            snippet_group = self.dict_snippets['snippet_groups'][index] 
            
            with open('trees_and_coref_links/snippet_trees/'+self.snippet_selection+'/'+str(index)+'.pickle','rb') as handle:
                snippet_trees = pickle.load(handle)
            
           
            for parse_tree in snippet_trees:
                try:
                    text_w_rels = identifier.analyze(parse_tree)
                    relations = re.findall(r'[A-Za-z]+#[0-9]#[A-Za-z0-1]+',text_w_rels)
                    if len(relations) != 0:
                        for rel in relations:
                            rel_arr = rel.split('#')
                            if rel_arr[2] != '0':
                                key = rel_arr[2].lower()
                                discourse_relations[key] = discourse_relations[key] + 1
                except Exception as err:
                    print(err,' occurred.')
                finally:
                    pass
          
            
            
            '''
            for sent in snippet_group:
                output = nlp.annotate(sent, properties={'annotators': 'parse','outputFormat': 'json','timeout': 800000})
                obj = json.loads(output)
                parse_tree =  obj['sentences'][0]['parse']
                text_w_rels = identifier.analyze(parse_tree)
                relations = re.findall(r'[A-Za-z]+#[0-9]#[A-Za-z0-1]+',text_w_rels)
                if len(relations) != 0:
                    for rel in relations:
                        rel_arr = rel.split('#')
                        if rel_arr[2] != '0':
                            key = rel_arr[2].lower()
                            discourse_relations[key] = discourse_relations[key] + 1
            '''
            
            for key in list(discourse_relations.keys()):
                dict_discourse_rel[key].append(discourse_relations[key])
        self.dict_discourse_rel = dict_discourse_rel
        return dict_discourse_rel
    
    def continuity(self):   
        # Run StanfortCoreNLP server before using this method.
        nlp = spacy.load('en_core_web_sm')
        dict_continuity = {'num_pron':[],'num_det':[],'avg_sim':[],'coref_links':[],'intra_links':[],'inter_links':[]}
        
        snippets = self.dict_snippets['snippets']
        for index,text in enumerate(tqdm(snippets,desc='Collecting Continuity features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            # number of pronouns and determiners
            doc = nlp(text)
            num_pron = 0
            num_det = 0
            for token in doc:
                if token.pos_ == 'PRON':
                    num_pron = num_pron + 1
                elif token.pos_== 'DET':
                    num_det = num_det + 1
            dict_continuity['num_pron'].append(num_pron)
            dict_continuity['num_det'].append(num_det)
            
            #similarity avg
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(self.dict_snippets['snippet_groups'][index])
            vectorizer.get_feature_names_out()
            result = X.toarray()
            sim_vals = []
            for i in range(result.shape[0]):
                if i+1 in range(result.shape[0]):
                    curr_sent = np.reshape(result[i],(-1,result[i].shape[0]))
                    next_sent = np.reshape(result[i+1],(-1,result[i+1].shape[0]))
                    sim_val = cosine_similarity(curr_sent,next_sent)
                    sim_vals.append(sim_val.ravel()[0])
            avg_sim = 0
            if len(sim_vals) > 0:
                avg_sim = sum(sim_vals)/len(sim_vals)
            dict_continuity['avg_sim'].append(avg_sim)
            
  
            #entity coreference links
            if self.snippet_selection == 'start':
                folder_path = 'trees_and_coref_links/coref_links/start_snippets'
            elif self.snippet_selection == 'end':
                folder_path = 'trees_and_coref_links/coref_links/end_snippets'
                
            file_name = str(index)+'.pickle'
            with open(os.path.join(folder_path,file_name),'rb') as handle:
                dict_coref = pickle.load(handle)
            
            #intra_links,inter_links = self.coreference_links(self.dict_snippets['snippets'][index])
            dict_continuity['intra_links'].append(dict_coref['intra_links'])
            dict_continuity['inter_links'].append(dict_coref['inter_links'])
            dict_continuity['coref_links'].append(dict_coref['coref_links'])
            
        self.dict_continuity = dict_continuity
        return dict_continuity
    
    def amount_of_detail(self,df_external):       # df_external is dataset used for generating specificity features
        '''
        feat_folder_path = 'features/test'
        feat_files = ['necd_features.pickle','polarity_features.pickle','sentence_length_features.pickle',
                      'specificity_features.pickle','syntactic_features.pickle','lm_features.pickle']
        
        feats = {}
        print('Collecting specificity features.')
        for i in feat_files:
            with open(os.path.join(feat_folder_path,i),'rb') as handle:
                dict_f = pickle.load(handle)
            feats.update(dict_f)
            
        #word feats
        with open(os.path.join(feat_folder_path,'word_features.pickle'),'rb') as handle:
            dict_f = pickle.load(handle)
            print()
            word_feats = dict_f['word_feat']
       
        df_feats = pd.DataFrame(feats)
        X = np.concatenate((df_feats.values,word_feats.todense()),axis = 1)
                
        with open('/Users/deepak/Desktop/thesis_pqai/specificity_model/models/instantiation/specificity_model.pickle',
                      'rb') as f:
            clf = pickle.load(f)
        print('Predicting sentence specificity.')
        y_pred = clf.predict_proba(np.asarray(X))
        y_pred_int = np.argmax(1*(y_pred > 0.5),axis=1)
                 
        
        all_snip_with_ids = []
        for index,group in enumerate(self.dict_snippets['snippet_groups']):
            list_el = []
            for sent in group:
                list_el.append([index,sent])
            all_snip_with_ids = all_snip_with_ids + list_el
        df_temp = pd.DataFrame(all_snip_with_ids,columns=['snippet_id','text'])
        df_temp['pred_label'] = y_pred_int
        print(df_temp.head(10))
           
        '''
        feats,df_temp = self.predict_specificity()
            
        
        dict_amount_of_detail = {'adjective_count':[],'adverb_count':[],'number_of_NEs':[],
                                 'avg_ne_length':[],'sents_wo_nes':[],'specificity_perc':[],'avg_word_sp':[]}
        print('Collecting amount of detail features.')
        snippets = self.dict_snippets['snippets']
        for index,text in enumerate(tqdm(snippets,desc='Collecting amount of detail features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            doc = self.nlp(text)
            adjective_count = 0
            adverb_count = 0

            # 1,2 = Number of adjectives and adverbs
            for token in doc:
                if token.pos_ == 'ADJ':
                    adjective_count = adjective_count + 1
                elif token.pos == 'ADV':
                    adverb_count = adverb_count + 1
            dict_amount_of_detail['adjective_count'].append(adjective_count)
            dict_amount_of_detail['adverb_count'].append(adverb_count)

            # 3. Number of named entities
            named_entities = doc.ents
            #named_entities = self.tag_ner(self.dict_snippets['snippets'][index])
            number_of_NEs = len(named_entities)
            dict_amount_of_detail['number_of_NEs'].append(number_of_NEs)

            # 4. Avg. length of NE's in words
            avg_ne_length = 0
            if len(named_entities) != 0:
                lengths = []
                for i in named_entities:
                    lengths.append(self.count_words(i.text))
                avg_ne_length = sum(lengths)/len(lengths)
            dict_amount_of_detail['avg_ne_length'].append(avg_ne_length)

            # 5. Number of sentences without NEs
            sents_wo_nes = 0
            for i in self.dict_snippets['snippet_groups'][index]:
                doc_indv_sent = self.nlp(i)
                ent = doc_indv_sent.ents
                if len(ent) == 0:
                    sents_wo_nes = sents_wo_nes + 1
            dict_amount_of_detail['sents_wo_nes'].append(sents_wo_nes)
            
            # 6. specificity features
            df_snippet = df_temp.loc[df_temp['snippet_id'] == index]
            #snippet_group = self.dict_snippets['snippet_groups'][index]
            #df_snippet_group = pd.DataFrame({'text':snippet_group,'labels':np.ones(len(snippet_group),dtype=int)})
            #fe = ExtractFeatures(df_pdtb = df_snippet_group, df_patent=df_external,state='test')
            #X = fe.extract_test_features()
            #with open('/Users/deepak/Desktop/thesis_pqai/specificity_model/models/instantiation/specificity_model.pickle',
            #          'rb') as f:
            #    clf = pickle.load(f)
            #y_pred = clf.predict_proba(np.asarray(X))
            #y_pred_int = np.argmax(1*(y_pred > 0.5),axis=1)
            
            ones = np.sum(df_snippet['pred_label'].values)
            #ones= y_pred_int.tolist().count(1)
            
            specificity_perc = (ones * 100)/df_snippet.shape[0]
            dict_amount_of_detail['specificity_perc'].append(specificity_perc)
            
            
            indexes = df_temp.index[df_temp['snippet_id'] == index]
            
            word_sp = np.array(feats['avg_noun_dist'])[indexes]
            
            #word_sp = fe.dict_specificity_features['avg_noun_dist']
            avg_word_sp = sum(word_sp)/len(word_sp)
            
            dict_amount_of_detail['avg_word_sp'].append(avg_word_sp)

        self.dict_amount_of_detail = dict_amount_of_detail
        
        return dict_amount_of_detail
        
    
    def syntactic_realization(self):
        #This method will be used only for reading the parse_tree files. If the trees are not generated already, generate them
        # using generate_trees() method from verbosity_helpers.SyntacticRealization -> TreeProductions class.
        
        
        dict_syntactic_realization = {'nps_prod_count':[],'non_nps_prod_count':[]}
        
        snippets = self.dict_snippets['snippets']
        for index,text in enumerate(tqdm(snippets,desc='Collecting production counts from parse trees.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            
            
            nps = []
            non_nps = []
            with open('trees_and_coref_links/snippet_trees/'+self.snippet_selection+'/'+str(index)+'.pickle','rb') as handle:
                arr_trees = pickle.load(handle)
            
            for parse_tree in arr_trees:
                t = Tree.fromstring(parse_tree)
                for prod in t.productions():
                    if prod.lhs().symbol() == 'NP':
                        nps.append(prod)
                    else:
                        non_nps.append(prod)
            
            nps_prod_count = []
            for prod in self.most_common_productions['freq_nps']:
                nps_prod_count.append(nps.count(prod[0]))
            dict_syntactic_realization['nps_prod_count'].append(nps_prod_count)
            
                
                
            non_nps_prod_count = []
            for prod in self.most_common_productions['freq_non_nps']:
                non_nps_prod_count.append(nps.count(prod[0]))
            dict_syntactic_realization['non_nps_prod_count'].append(non_nps_prod_count)
     
            
        self.dict_syntactic_realization = dict_syntactic_realization
        return self.dict_syntactic_realization
    
    def compression_likelihood(self):
        obj_comp_like = CompreesionLikelihood()
        top25 =  obj_comp_like.most_commonly_deleted_productions()
        
        dict_compression_likelihood = {}
        for i in top25:
            dict_compression_likelihood.update({i:[]})
        dict_compression_likelihood.update({'sum_cl':[]})
        dict_compression_likelihood.update({'avg_cl':[]})
        dict_compression_likelihood.update({'prod_cl':[]})
        dict_compression_likelihood.update({'pp_cl':[]})

        
        snippets = self.dict_snippets['snippets']
        for index,text in enumerate(tqdm(snippets,desc='Collecting compression likelihood features.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            
            snippet_path = 'trees_and_coref_links/snippet_trees/'+self.snippet_selection+'/'+str(index)+'.pickle'
            feats = obj_comp_like.features_for_snippet(snippet_path)
            for key in list(feats.keys()):
                dict_compression_likelihood[key].append(feats[key])
            
        self.dict_compression_likelihood = dict_compression_likelihood
        return dict_compression_likelihood
