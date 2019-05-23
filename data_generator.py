# -*- coding: utf-8 -*-
"""
generate train and valid/test data

"""

import ast
import re
import json
import os

from setting import Path

class Generator:
    def __init__(self, paras):
        self.__paras = paras
        self.__data_dir = self.__paras.dataset_path
        
        # if a word/tree node has frequency less than min_vocabulary_count
        # we replace it to 'unknown', generally 'unknown' belongs to string
        self.__min_vocabulary_count = self.__paras.min_vocabulary_count
        self.__nl_vocabulary = {'unknwon' : 0}
        self.__tree_nodes_vocabulary = {'unknwon' : 0, '<END_Node>' : 1, '<Empty_Node>' : 2}
        
        # statistical data
        self.__nl_max_length = 0
        self.__nl_length_sum = 0.0
        self.__tree_nodes_max_num = 0
        self.__tree_nodes_sum = 0.0
        self.__semantic_unit_max_num = 0
        self.__data_num = 0.0
        
    
        ''' process and generate the train and test data '''
        train_data_provider = Generator.__Data_provider(self.__data_dir, 'train', self.__paras)
        test_data_provider = Generator.__Data_provider(self.__data_dir, 'test', self.__paras)
        
        # register vocabulary
        self.__register_ids_to_vocabulary([train_data_provider, test_data_provider])
        train_data, train_data_for_read = self.__process_each(train_data_provider)
        test_data, test_data_for_read = self.__process_each(test_data_provider)
        
        if not os.path.exists(self.__data_dir + Path.GENERATED_PATH):
            os.mkdirs(self.__data_dir + Path.GENERATED_PATH)
        self.__write_nl_vocabulary()
        self.__write_tree_nodes_vocabulary()
        self.__write_data(train_data, self.__data_dir + Path.TRAIN_DATA_PATH)
        self.__write_data(train_data_for_read, self.__data_dir + Path.TRAIN_DATA_PATH + '_for_read')
        self.__write_data(test_data, self.__data_dir + Path.TEST_DATA_PATH)
        self.__write_data(test_data_for_read, self.__data_dir + Path.TEST_DATA_PATH + '_for_read')
        
        self.__write_statistical_data()
        
    
    '''
    '''
    class __Data_provider:
        def __init__(self, dataset_path, train_or_test, paras):
            ''' CONALA '''
            if (dataset_path == Path.CONALA_PATH):
                if (train_or_test == 'train'):
                    path = dataset_path + 'conala-train.json'
                else:
                    path = dataset_path + 'conala-test.json'
                
                with open(path, 'r') as f:
                    data_read_conala = f.read()
                # there are some null descriptions in data , so we need define the 'null' variable before eval
                null = 'null'
                self.__data_read_conala = eval(data_read_conala)                
                
                self.data_iter = self.__data_iter_conala
                return
            
            ''' HS '''
            if (dataset_path == Path.HS_PATH):
                if (train_or_test == 'train'):
                    path_in = dataset_path + 'train_hs.in'
                    path_out = dataset_path + 'train_hs.out'
                else:
                    path_in = dataset_path + 'dev_hs.in'
                    path_out = dataset_path + 'dev_hs.out'
                with open(path_in, 'r', encoding='utf-8') as f_in, open(path_out, 'r', encoding='utf-8') as f_out:
                    data_read_in = f_in.readlines()
                    data_read_out = f_out.readlines()
                    num = len(data_read_in)
                    self.__data_read_hs = []
                    for i in range(num):
                        if (data_read_in[i] != ''):
                            self.__data_read_hs.append([data_read_in[i], data_read_out[i]])
                    
                self.data_iter = self.__data_iter_hs
                return
                
        def __data_iter_conala(self):
            for data_unit in self.__data_read_conala:
                description = data_unit['rewritten_intent']
                # skip the null description
                if (description == 'null') : continue
                # delete the ' " ` 
                description = re.sub('[\'\"`]', '', description).strip()
                description = description.split(' ')
                ast_root = ast.parse(data_unit['snippet'])
                yield description, ast_root 
        def __data_iter_hs(self):
            for description, code in self.__data_read_hs:
                description = re.sub(r'<b>|</b>|\.|,|;|\$|#|<i>|</i>|\(|\)', '', description).strip()
                description = description.split(' ')
                
                code = code.replace('§', '\n')
                code = code.replace('\ ', '')
                ast_root = ast.parse(code)
                yield description, ast_root
        
        
    ''' 
    process train or test data 
    @result :[
       [[description],
        [node_list],
        [node_parent_list],
        [node_grandparent_list],
        [semantic_unit_list],
        [semantic_unit_children_list],
        [correct_prediction]],
        ...
    ]
    '''
    def __process_each(self, data_provider):
        
        result = []
        result_for_read = []
        for description, ast_root in data_provider.data_iter():
            self.__data_num += 1
            
            ''' description '''
            # register description
            description_ids = self.__get_ids_from_nl_vocabulary(description)
            
            #
            self.__nl_length_sum += len(description)
            if len(description) > self.__nl_max_length : self.__nl_max_length = len(description)
            
            ''' ast '''
            node_list = []
            node_parent_list = []
            node_grandparent_list = []
            traceable_node_list = []
            parent = '<Empty_Node>'
            grandparent = '<Empty_Node>'
            terminal = '<END_Node>'
           
            self.__process_node(ast_root, parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            
            #
            self.__tree_nodes_sum += len(node_list)
            if len(node_list) > self.__tree_nodes_max_num : self.__tree_nodes_max_num = len(node_list)
            
            ''' process semantic unit & fill in the result '''
            # each node prediction refers to a train/test data
            data_num = len(node_list)
            
            for count in range(data_num):
                # copy the list
                node_list = node_list[:]
                node_parent_list = node_parent_list[:]
                node_grandparent_list = node_grandparent_list[:]
                traceable_node_list = traceable_node_list[:]
                # get the correct prediciton and delete the last node in the lists
                if (count == 0):
                    correct_prediction = terminal
                else:
                    # delete the last node
                    correct_prediction = node_list.pop()
                    node_parent_list.pop()
                    node_grandparent_list.pop()
                    temp = traceable_node_list.pop()
                    while (temp == '{' or temp == '}'):
                        temp = traceable_node_list.pop()
                
                # process the semantic unit
                semantic_unit_list, semantic_unit_children_list = Generator.process_semantic_unit(traceable_node_list, self.__paras)   
                
                if (len(semantic_unit_list) > self.__semantic_unit_max_num): self.__semantic_unit_max_num = len(semantic_unit_list)
                
                # fill in the result(for read)
                data_unit = []
                data_unit.append(description)
                data_unit.append(node_list)
                data_unit.append(node_parent_list)
                data_unit.append(node_grandparent_list)
                data_unit.append(semantic_unit_list)
                data_unit.append(semantic_unit_children_list)
                data_unit.append([correct_prediction])
                result_for_read.append(data_unit)
                
                # fill in the result(ids)
                data_unit_ids = []
                data_unit_ids.append(description_ids)
                data_unit_ids.append(self.__get_ids_from_tree_nodes_vocabulary(node_list))
                data_unit_ids.append(self.__get_ids_from_tree_nodes_vocabulary(node_parent_list))
                data_unit_ids.append(self.__get_ids_from_tree_nodes_vocabulary(node_grandparent_list))
                data_unit_ids.append(self.__get_ids_from_tree_nodes_vocabulary(semantic_unit_list))               
                semantic_unit_children_list_ids = []
                for children in semantic_unit_children_list:
                    semantic_unit_children_list_ids.append(self.__get_ids_from_tree_nodes_vocabulary(children))
                data_unit_ids.append(semantic_unit_children_list_ids)
                data_unit_ids.append(self.__get_ids_from_tree_nodes_vocabulary(correct_prediction))
                result.append(data_unit_ids)
                
        return result, result_for_read

    '''
    recursively append node and its parent&grandparent to list
    @node, parent, grandparent : current node and its parent node(str) and grandparent node(str)
    @node_list, node_parent_list, node_grandparent_list : list of str name of node
    @traceable_node_list : traceable list with '{' and '}' mark the block, and a '<data_point>' before to mark the data node(str/int/float/bool)
    '''
    def __process_node(self, node, parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list):
        # as a data point
        if (isinstance(node, str) or (isinstance(node, int)) or (isinstance(node, float)) or (isinstance(node, bool))):
            # append <data_point> before appending data node
            traceable_node_list.append('<data_point>')
            self.__appends(str(node), parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            return
        # as a list
        if (isinstance(node, list)):
            if (len(node) == 0):
                self.__appends('<Empty_List>', parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            else:
                self.__appends('<List>', parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
                traceable_node_list.append('{')
                for child in node:
                    self.__process_node(child, '<List>', parent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
                traceable_node_list.append('}')
            return
        # as a AST node
        if (isinstance(node, ast.AST)):
            name = 'ast.' + node.__class__.__name__
            self.__appends(name, parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            traceable_node_list.append('{')
            for child_name, child_field in ast.iter_fields(node):
                self.__process_node(child_field, name, parent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            traceable_node_list.append('}')
            return
        
        if (node == None):
            self.__appends('<None_Node>', parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            return
        
        print('error: type ' + str(type(node)))
        
    ''' support method for __process_node, which append the str data to list '''
    def __appends(self, v1, v2, v3, l1, l2, l3, traceable_l1):
        l1.append(v1)
        traceable_l1.append(v1)
        l2.append(v2)
        l3.append(v3)    
    
    '''
    get semantic_unit_list and semantic_unit_children_list by processing traceable_node_list
    '''
    @staticmethod
    def process_semantic_unit(traceable_node_list, paras):
        semantic_unit_list = []
        semantic_unit_children_list = []      
        for count in range(len(traceable_node_list)):
            # reach a semantic unit
            node = traceable_node_list[count]
            if not (Generator.__is_semantic_node(node)):
                continue
            
            semantic_unit_list.append(node)
            
            # children with their score
            children_with_score = []
            # search children
            depth = 0
            sub_count = count + 1
            is_data_point = False
            # a semantic unit must be followed by a '{' 
            depth += 1
            sub_count += 1
            # loop while depth back to 0 or list through over
            while (depth > 0 and sub_count < len(traceable_node_list)):
                next_node = traceable_node_list[sub_count]
                if (next_node == '{'):
                    depth += 1
                elif (next_node == '}'):
                    depth -= 1
                elif (next_node == '<data_point>'):
                    is_data_point = True
                else:
                    children_with_score.append([next_node, Generator.__semantic_child_score(is_data_point, depth)])
                    is_data_point = False
                sub_count += 1    
            # reduce children by k max, and we need to keep the order of node data
            children = []
            if (len(children_with_score) <= paras.semantic_unit_children_num):
                for c in children_with_score:
                    children.append(c[0])
            else:
                children_with_score_copy = children_with_score[:]
                children_with_score_copy.sort(key=lambda a: a[1])
                k_max_score = children_with_score_copy[paras.semantic_unit_children_num - 1][1]
                children_num = paras.semantic_unit_children_num
                for c in children_with_score:
                    if (c[1] >= k_max_score):
                        children.append(c[0])
                        children_num -= 1
                        if (children_num == 0): 
                            break
            
            semantic_unit_children_list.append(children)
        
        return semantic_unit_list, semantic_unit_children_list
    
    ''' @return : whether the node is a semantic unit '''
    @staticmethod
    def __is_semantic_node(node):
        return (node == 'ast.Call' or
                node == 'ast.Attribute' or
                node == 'ast.Assign' or
                node == 'ast.AugAssign' or
                node == 'ast.While' or
                node == 'ast.If')
    ''' 
    score the child's contribution of semantic information
    '''
    @staticmethod
    def __semantic_child_score(is_data_point, depth):
        reward = 2.5 if is_data_point else 0.0
        return reward - depth
    
    
    '''
    register each word/tree node with v[word] = len(v)
    if the word/tree node 's frequency less than self.__min_vocabulary_count, it won't be registered
    '''
    def __register_ids_to_vocabulary(self, data_providers):
        nl_vocabulary_with_count = {}
        tree_nodes_vocabulary_with_count = {}
        for data_provider in data_providers:
            for description, ast_root in data_provider.data_iter():
                ''' description '''
                for word in description:
                    if word not in nl_vocabulary_with_count:
                        nl_vocabulary_with_count[word] = 1
                    else:
                        nl_vocabulary_with_count[word] += 1
                        
                ''' ast '''
                nodes = []
                self.__process_node(ast_root, '', '', nodes, [], [], [])
                
                for node in nodes:
                    if node not in tree_nodes_vocabulary_with_count:
                        tree_nodes_vocabulary_with_count[node] = 1
                    else:
                        tree_nodes_vocabulary_with_count[node] += 1
                
        for word, count in nl_vocabulary_with_count.items():
            if (count >= self.__min_vocabulary_count): self.__nl_vocabulary[word] = len(self.__nl_vocabulary)
        for node, count in tree_nodes_vocabulary_with_count.items():
            if (count >= self.__min_vocabulary_count): self.__tree_nodes_vocabulary[node] = len(self.__tree_nodes_vocabulary)
    

    def __get_ids_from_nl_vocabulary(self, words):
        return Generator.get_ids_from_nl_vocabulary(words, self.__nl_vocabulary)
    '''
    get id from nl vocabulary, if not in vocabulary return 'unknown':0
    '''
    @staticmethod
    def get_ids_from_nl_vocabulary(words, nl_vocabulary):
        if not (isinstance(words, list)): words = [words]
        ids = []
        for word in words:
            if (word not in nl_vocabulary):
                ids.append(0)
            else:
                ids.append(nl_vocabulary[word])
        return ids    
    def __get_ids_from_tree_nodes_vocabulary(self, nodes):
        return Generator.get_ids_from_tree_nodes_vocabulary(nodes, self.__tree_nodes_vocabulary)
    '''
    get id from tree nodes vocabulary, if not in vocabulary return 'unknown':0
    '''    
    @staticmethod
    def get_ids_from_tree_nodes_vocabulary(nodes, tree_nodes_vocabulary):
        if not (isinstance(nodes, list)): nodes = [nodes]
        ids = []
        for node in nodes:
            if (node not in tree_nodes_vocabulary):
                ids.append(0)
            else:
                ids.append(tree_nodes_vocabulary[node])
        return ids
    

       
    def __write_nl_vocabulary(self):
        path = self.__data_dir + Path.NL_VOCABULARY_PATH
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__nl_vocabulary, indent=1))
        
    def __write_tree_nodes_vocabulary(self):
        path = self.__data_dir + Path.TREE_NODES_VOCABULARY_PATH
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__tree_nodes_vocabulary, indent=1))

        
    ''' write one(string) for read and one(ids) for model '''
    def __write_data(self, data, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('# -*- coding: utf-8 -*-\n')
            f.write('[\n')
            for count in range(len(data)):
                paragraph = data[count]
                f.write(' [')
                f.write(str(paragraph[0])) # description
                f.write(',\n  ')
                f.write(str(paragraph[1])) # node list
                f.write(',\n  ')
                f.write(str(paragraph[2])) # node parent list
                f.write(',\n  ')
                f.write(str(paragraph[3])) # node grandparent list
                f.write(',\n  ')
                f.write(str(paragraph[4])) # semantic unit list
                f.write(',\n  [')
                for children_count in range(len(paragraph[5])): # semantic unit children list
                    f.write(str(paragraph[5][children_count]))
                    if (children_count != len(paragraph[5]) - 1):
                        f.write(',\n   ')
                f.write('],\n  ')
                f.write(str(paragraph[6])) # correct prediction
                f.write(']')
                if (count != len(data) - 1):
                    f.write(',\n\n')
            f.write('\n]')
    
    '''  '''
    def __write_statistical_data(self):
        with open(self.__data_dir + Path.STATISTICS_PATH, 'w', encoding='utf-8') as f:
            f.write('nl vocabulary size: ' + str(len(self.__nl_vocabulary)) + '\n')
            f.write('tree nodes vocabulary size: ' + str(len(self.__tree_nodes_vocabulary)) + '\n')
            f.write('nl average length: ' + str(self.__nl_length_sum / self.__data_num) + '\n')
            f.write('nl max length: ' + str(self.__nl_max_length) + '\n')
            f.write('tree nodes average num: ' + str(self.__tree_nodes_sum / self.__data_num) + '\n')
            f.write('tree nodes max num: ' + str(self.__tree_nodes_max_num) + '\n')
            f.write('semantic unit max num: ' + str(self.__semantic_unit_max_num) + '\n')
            
            
if (__name__ == '__main__'):
    from setting import Parameters
#    handle = Generator(Parameters.get_hs_paras())
    handle = Generator(Parameters.get_conala_paras())