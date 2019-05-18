# -*- coding: utf-8 -*-
"""
generate train & valid/test data

"""

import ast
import re
import json

class Generator:
    HS_PATH = 'data/hearthstone'
    CONALA_PATH = 'data/conala-corpus/'
    def __init__(self, dataset_path):
        if (dataset_path != Generator.HS_PATH and dataset_path != Generator.CONALA_PATH): raise Exception('Wrong Path')
        self.__data_dir = dataset_path
        
        self.__semantic_unit_children_num = 3
        # if a word/tree node has frequency less than min_vocabulary_count
        # we replace it to 'unknown', generally 'unknown' belongs to string
        self.__min_vocabulary_count = 5
        self.__nl_vocabulary = {'unknwon' : 0}
        self.__tree_nodes_vocabulary = {'unknwon' : 0, '<END_Node>' : 1}
        
        # statistical data
        self.__nl_max_length = 0
        self.__nl_length_sum = 0.0
        self.__tree_nodes_max_num = 0
        self.__tree_nodes_sum = 0.0
        self.__semantic_unit_max_num = 0
        self.__data_num = 0.0
        
    
        ''' process and generate the train and test data '''
        train_data_provider = Generator.__Data_provider(self.__data_dir + 'conala-train.json')
        test_data_provider = Generator.__Data_provider(self.__data_dir + 'conala-test.json')
        # register vocabulary
        self.__register_ids_to_vocabulary([train_data_provider, test_data_provider])
        train_data, train_data_for_read = self.__process_each(train_data_provider)
        test_data, test_data_for_read = self.__process_each(test_data_provider)
        
        self.__write_nl_vocabulary()
        self.__write_tree_nodes_vocabulary()
        self.__write_data(train_data, self.__data_dir + 'train_data')
        self.__write_data(train_data_for_read, self.__data_dir + 'train_data_for_read')
        self.__write_data(test_data, self.__data_dir + 'test_data')
        self.__write_data(test_data_for_read, self.__data_dir + 'test_data_for_read')
        
        self.__write_statistical_data()
        
    
    '''
    '''
    class __Data_provider:
        def __init__(self, path):
            with open(path, 'r') as f:
                data_read = f.read()
            # there are some null descriptions in data , so we need define the 'null' variable before eval
            null = 'null'
            self.__data_read = eval(data_read)
            
            '''
            @description : list of word
            @ast_root : root of ast
            '''
            self.data_iter = self.__data_iter_conala
        
#        def __ini_conala
        def __data_iter_conala(self):
            for data_unit in self.__data_read:
                description = data_unit['rewritten_intent']
                # skip the null description
                if (description == 'null') : continue
                # delete the ' " ` 
                description = re.sub('[\'\"`]', '', description).strip()   
                description = description.split(' ')
                ast_root = ast.parse(data_unit['snippet'])
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
                semantic_unit_list, semantic_unit_children_list = self.__process_semantic_unit(traceable_node_list)   
                
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
            self.__appends(node.__class__.__name__, parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            traceable_node_list.append('{')
            for child_name, child_field in ast.iter_fields(node):
                self.__process_node(child_field, node.__class__.__name__, parent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            traceable_node_list.append('}')
            return
        
        if (node == None):
            self.__appends('<Empty_Node>', parent, grandparent, node_list, node_parent_list, node_grandparent_list, traceable_node_list)
            return
        
        print('error: type ' + str(type(node)))
        
    ''' support method for __process_node, which append the str data to list '''
    def __appends(self, v1, v2, v3, l1, l2, l3, traceable_l1):
        l1.append(v1)
        traceable_l1.append(v1)
        l2.append(v2)
        l3.append(v3)    
    
    '''
    '''
    def __process_semantic_unit(self, traceable_node_list):
        semantic_unit_list = []
        semantic_unit_children_list = []      
        for count in range(len(traceable_node_list)):
            # reach a semantic unit
            node = traceable_node_list[count]
            if not (self.__is_semantic_node(node)):
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
                    children_with_score.append([next_node, self.__semantic_child_score(is_data_point, depth)])
                    is_data_point = False
                sub_count += 1    
            # reduce children by k max, and we need to keep the order of node data
            children = []
            if (len(children_with_score) <= self.__semantic_unit_children_num):
                for c in children_with_score:
                    children.append(c[0])
            else:
                children_with_score_copy = children_with_score[:]
                children_with_score_copy.sort(key=lambda a: a[1])
                k_max_score = children_with_score_copy[self.__semantic_unit_children_num - 1][1]
                children_num = self.__semantic_unit_children_num
                for c in children_with_score:
                    if (c[1] >= k_max_score):
                        children.append(c[0])
                        children_num -= 1
                        if (children_num == 0): 
                            break
            
            semantic_unit_children_list.append(children)
        
        return semantic_unit_list, semantic_unit_children_list
    
    ''' @return : whether the node is a semantic unit '''
    def __is_semantic_node(self, node):
        return (node == 'Call' or
                node == 'Attribute' or
                node == 'Assign' or
                node == 'AugAssign' or
                node == 'While' or
                node == 'If')
    ''' 
    score the child's contribution of semantic information
    '''
    def __semantic_child_score(self, is_data_point, depth):
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
    
    '''
    get id from nl vocabulary, if not in vocabulary return 'unknown':0
    '''
    def __get_ids_from_nl_vocabulary(self, words):
        if not (isinstance(words, list)): words = [words]
        ids = []
        for word in words:
            if (word not in self.__nl_vocabulary):
                ids.append(0)
            else:
                ids.append(self.__nl_vocabulary[word])
        return ids
    '''
    get id from tree nodes vocabulary, if not in vocabulary return 'unknown':0
    '''
    def __get_ids_from_tree_nodes_vocabulary(self, nodes):
        if not (isinstance(nodes, list)): nodes = [nodes]
        ids = []
        for node in nodes:
            if (node not in self.__tree_nodes_vocabulary):
                ids.append(0)
            else:
                ids.append(self.__tree_nodes_vocabulary[node])
        return ids


       
    def __write_nl_vocabulary(self):
        path = self.__data_dir + 'nl_vocabulary'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.__nl_vocabulary, indent=1))
        
    def __write_tree_nodes_vocabulary(self):
        path = self.__data_dir + 'tree_nodes_vocabulary'
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
        with open(self.__data_dir + 'statistics', 'w', encoding='utf-8') as f:
            f.write('nl vocabulary size: ' + str(len(self.__nl_vocabulary)) + '\n')
            f.write('tree nodes vocabulary size: ' + str(len(self.__tree_nodes_vocabulary)) + '\n')
            f.write('nl average length: ' + str(self.__nl_length_sum / self.__data_num) + '\n')
            f.write('nl max length: ' + str(self.__nl_max_length) + '\n')
            f.write('tree nodes average num: ' + str(self.__tree_nodes_sum / self.__data_num) + '\n')
            f.write('tree nodes max num: ' + str(self.__tree_nodes_max_num) + '\n')
            f.write('semantic unit max num: ' + str(self.__semantic_unit_max_num) + '\n')
            
            

#handle = Generator(Generator.HS_PATH)
handle = Generator(Generator.CONALA_PATH)