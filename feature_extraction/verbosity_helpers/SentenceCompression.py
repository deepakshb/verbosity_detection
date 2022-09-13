import pickle
import os
from nltk import Tree
from collections import Counter
from tqdm import tqdm
import math


class CompreesionLikelihood:
    def __init__(self):
        self.folder_path = 'trees_and_coref_links/sent_comp_trees'
        self.file_names = os.listdir(self.folder_path)
    
    def read_productions_and_identify_deletion(self):
        corpus_productions = []
        corpus_prod_for_deletion = []

        for index,name in enumerate(tqdm(self.file_names,desc='Collecting productions undergoing deletion.',bar_format='{l_bar}{bar} [{n_fmt}/{total_fmt}]')):
            with open(os.path.join(self.folder_path,name),'rb') as handle:
                dict_sent_comp = pickle.load(handle)

            #Productions from abstracts
            productions = []
            for tree in dict_sent_comp['abstract_tree']:
                t = Tree.fromstring(tree)
                productions = productions + t.productions()

            #Productions from the title
            tt = Tree.fromstring(dict_sent_comp['title_tree'])
            title_prods = tt.productions()

            #production with non-terminals from abstracts and title
            prod_with_non_terminals = [prod for prod in productions if prod.is_nonlexical()]
            title_prod_with_non_terminals = [prod for prod in title_prods if prod.is_nonlexical()]

            #extract nodes from title productions
            all_title_nodes = []
            for title_prod in title_prod_with_non_terminals:
                lhs = title_prod.lhs().symbol()
                rhs = [i.symbol() for i in list(title_prod.rhs())]
                all_title_nodes.append(lhs)
                all_title_nodes = all_title_nodes + rhs


            # A production (LHS → RHS) is said to undergo deletion when either the LHS node or any of the nodes in the 
            # RHS do not appear in the compressed sentence.
            prod_for_deletion = []
            for prod in prod_with_non_terminals:
                lhs = prod.lhs().symbol()
                rhs = [i.symbol() for i in list(prod.rhs())]
                if lhs not in all_title_nodes:
                    prod_for_deletion.append(prod)
                else:
                    for symbol in rhs:
                        if symbol not in all_title_nodes:
                            prod_for_deletion.append(prod)
                            break

            corpus_prod_for_deletion = corpus_prod_for_deletion + prod_for_deletion
            corpus_productions  = corpus_productions + prod_with_non_terminals
        return corpus_prod_for_deletion,corpus_productions
    
    def most_commonly_deleted_productions(self):
        
        del_prod_file_path = 'most_common_deleted.pickle'
        dict_deletion = {}
        if os.path.exists(del_prod_file_path):
            print('Most deleted 25 productions are available.')
            with open('most_common_deleted.pickle', 'rb') as handle:
                dict_deletion = pickle.load(handle)
            self.top_25_prod = dict_deletion['top_25']
            self.dict_del_prob = dict_deletion['del_prob']
        else:
            print('Most deleted 25 productions are not available. Identifying them.')
            corpus_prod_for_deletion,corpus_productions = self.read_productions_and_identify_deletion()
            dict_deletion_counts = dict(Counter(corpus_prod_for_deletion))
            dict_prod_occurance_counts = dict(Counter(corpus_productions))

            self.dict_del_prob = {}
            dict_del_score = {}

            for key in list(dict_deletion_counts.keys()):
                del_count = dict_deletion_counts[key]
                total_count = dict_prod_occurance_counts[key]
                #print(del_count,' | ',total_count)
                prob = del_count/total_count
                self.dict_del_prob[key] = prob
                del_score = prob * math.log(total_count)
                dict_del_score[key] = del_score

            list_top_deleted = sorted(dict_del_score.items(), key=lambda item: item[1],reverse=True)[:25]
            self.top_25_prod = [i[0] for i in list_top_deleted]
            
            dict_deletion['top_25'] = self.top_25_prod
            dict_deletion['del_prob'] = self.dict_del_prob
            with open('most_common_deleted.pickle', 'wb') as handle:
                pickle.dump(dict_deletion, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.top_25_prod
    
    def features_for_snippet(self,snippet_tree_path):
        #Run most_commonly_deleted_productions() method once before using this method
        
        with open(snippet_tree_path,'rb') as handle:
            trees = pickle.load(handle)
        
        # Reading productions from trees
        productions = []
        for parse_tree in trees:
            t = Tree.fromstring(parse_tree)
            productions = productions + t.productions()
        
        #Count occurance of top25 productions in snippet productions
        feats = {}
        for prod in self.top_25_prod:
            feats.update({prod:productions.count(prod)})
                
        
        # Sum, average,product and perplexity of deleting probabilities
        prod_with_non_terminals = [prod for prod in productions if prod.is_nonlexical()]
        product = 1
        sum_cl = 0
        count = 0
        for prod in prod_with_non_terminals:
            if prod in list(self.dict_del_prob.keys()):
                count += 1
                prob = self.dict_del_prob[prod]
                product *= prob
                sum_cl += prob
        avg = 0
        if count > 0:
            avg = sum_cl/count
        perplexity = 0
        if count > 0 and product > 0:
            perplexity = product **(-1/count)
        
        # append Sum, average, product and perplexity of deleting probabilities as features
        feats.update({'sum_cl':sum_cl})
        feats.update({'avg_cl':avg})
        feats.update({'prod_cl':product})
        feats.update({'pp_cl':perplexity})
          
        return feats
        
        
        