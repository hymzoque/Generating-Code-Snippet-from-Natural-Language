# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import os

from setting import Path

class Data:
    def __init__(self, paras_list):
        data_dir = paras_list[0].dataset_path
        self.__train_batch_size = paras_list[0].train_batch_size
        
        suffixes = ['_ast_nodes', '_functions', '_variables', '_values']
        
        self.__train_data_ast_nodes = self.__train_data_process(data_dir + Path.TRAIN_DATA_PATH + suffixes[0], paras_list[0])
        self.__train_data_functions_name = self.__train_data_process(data_dir + Path.TRAIN_DATA_PATH + suffixes[1], paras_list[1])
        self.__train_data_variables_name = self.__train_data_process(data_dir + Path.TRAIN_DATA_PATH + suffixes[2], paras_list[2])
        self.__train_data_values = self.__train_data_process(data_dir + Path.TRAIN_DATA_PATH + suffixes[3], paras_list[3])
        
        self.__valid_batches_ast_nodes = self.__valid_data_process(data_dir + Path.TEST_DATA_PATH + suffixes[0], paras_list[0])
        self.__valid_batches_functions_name = self.__valid_data_process(data_dir + Path.TEST_DATA_PATH + suffixes[1], paras_list[1])
        self.__valid_batches_variables_name = self.__valid_data_process(data_dir + Path.TEST_DATA_PATH + suffixes[2], paras_list[2])
        self.__valid_batches_values = self.__valid_data_process(data_dir + Path.TEST_DATA_PATH + suffixes[3], paras_list[3])
        
        self.__unbalance_class_weights_ast_nodes = self.__read_unbalance_class_weights(data_dir + Path.UNBALANCE_CLASS_WEIGHTS_PATH + suffixes[0])
        self.__unbalance_class_weights_functions_name = self.__read_unbalance_class_weights(data_dir + Path.UNBALANCE_CLASS_WEIGHTS_PATH + suffixes[1])
        self.__unbalance_class_weights_variables_name = self.__read_unbalance_class_weights(data_dir + Path.UNBALANCE_CLASS_WEIGHTS_PATH + suffixes[2])
        self.__unbalance_class_weights_values = self.__read_unbalance_class_weights(data_dir + Path.UNBALANCE_CLASS_WEIGHTS_PATH + suffixes[3])
        
    '''
    data form
    [
     [
      [input_NL],
      [input_ast_nodes],
      [input_ast_parent_nodes],
      [input_ast_grandparent_nodes],
      [input_semantic_units],
      [input_children_of_semantic_units],
      [correct_output] 
     ],
     ...
    ]
    
    '''
    def __train_data_process(self, path, paras):
        train_data = []
        i = 0
        while (os.path.exists(path + str(i))):
            with open(path + str(i), 'r', encoding='utf-8') as f:
                train_data.extend(eval(f.read()))
            i += 1
        return self.__data_process(train_data, paras)
        
    '''
    data form same with __train_data_process
    '''
    def __valid_data_process(self, path, paras):
        test_data = []
        i = 0
        while (os.path.exists(path + str(i))):
            with open(path + str(i), 'r', encoding='utf-8') as f:
                test_data.extend(eval(f.read()))
            i += 1
        valid_data = self.__data_process(test_data, paras)    
        
        d0 = valid_data[0]
        d1 = valid_data[1]
        d2 = valid_data[2]
        d3 = valid_data[3]
        d4 = valid_data[4]
        d5 = valid_data[5]
        d6 = valid_data[6]        
        
        data_num = int(d0.shape[0])
        batch_size = paras.valid_batch_size
        batch_num = int(data_num / batch_size)
        
        valid_batches = []    
        
        for i in range(batch_num):
            valid_batches.append([
                    d0[i*batch_size : (i+1)*batch_size, :],
                    d1[i*batch_size : (i+1)*batch_size, :],
                    d2[i*batch_size : (i+1)*batch_size, :],
                    d3[i*batch_size : (i+1)*batch_size, :],
                    d4[i*batch_size : (i+1)*batch_size, :],
                    d5[i*batch_size : (i+1)*batch_size, :],
                    d6[i*batch_size : (i+1)*batch_size, :]
                    ])            
        
        return valid_batches
    
    
    ''' 
    fill 0 in spare place
    '''
    def __data_process(self, data, paras):
        data_num = len(data)
        d0 = np.zeros([data_num, paras.nl_len])
        d1 = np.zeros([data_num, paras.tree_len])
        d2 = np.zeros([data_num, paras.tree_len])
        d3 = np.zeros([data_num, paras.tree_len])
        d4 = np.zeros([data_num, paras.semantic_units_len])
        d5 = np.zeros([data_num, paras.semantic_units_len, paras.semantic_unit_children_num])
        d6 = np.zeros([data_num, paras.correct_predict_class_num])
        
        for i in range(data_num):
            data_point = data[i]
            
            nl = data_point[0]
            for j in range(len(nl)):
                d0[i][j] = nl[j]
            
            node = data_point[1]
            for j in range(len(node)):
                d1[i][j] = node[j]
            
            parent = data_point[2]
            for j in range(len(parent)):
                d2[i][j] = parent[j]
                
            grandparent = data_point[3]
            for j in range(len(grandparent)):
                d3[i][j] = grandparent[j]
            
            semantic = data_point[4]
            for j in range(len(semantic)):
                d4[i][j] = semantic[j]
                
            semantic_children = data_point[5]
            if (len(semantic_children) > 0):
                for j in range(len(semantic_children)):
                    for k in range(len(semantic_children[j])):
                        d5[i][j][k] = semantic_children[j][k]
            
            correct = data_point[6][0]
            d6[i][correct] = 1
        
        return [d0, d1, d2, d3, d4, d5, d6]

    
    
    '''
    shuffle train data
    @return batch_num x [
        [batch_size x input_NL],
        [batch_size x input_ast_nodes],
        [batch_size x input_ast_parent_nodes],
        [batch_size x input_ast_grandparent_nodes],
        [batch_size x input_semantic_units],
        [batch_size x input_children_of_semantic_units],
        [batch_size x correct_output],
    ]
    '''
    def __get_train_batches(self, train_data):
        d0 = train_data[0]
        d1 = train_data[1]
        d2 = train_data[2]
        d3 = train_data[3]
        d4 = train_data[4]
        d5 = train_data[5]
        d6 = train_data[6]
        
        data_num = int(d0.shape[0])
        shuffle = np.random.permutation(range(data_num))
        d0_shuffle = d0[shuffle]
        d1_shuffle = d1[shuffle]
        d2_shuffle = d2[shuffle]
        d3_shuffle = d3[shuffle]
        d4_shuffle = d4[shuffle]
        d5_shuffle = d5[shuffle]
        d6_shuffle = d6[shuffle]
        
        batch_size = self.__train_batch_size
        batch_num = int(data_num / batch_size)
        
        train_batches = []
        # fill in the batches
        for i in range(batch_num):
            train_batches.append([
                    d0_shuffle[i*batch_size : (i+1)*batch_size, :],
                    d1_shuffle[i*batch_size : (i+1)*batch_size, :],
                    d2_shuffle[i*batch_size : (i+1)*batch_size, :],
                    d3_shuffle[i*batch_size : (i+1)*batch_size, :],
                    d4_shuffle[i*batch_size : (i+1)*batch_size, :],
                    d5_shuffle[i*batch_size : (i+1)*batch_size, :],
                    d6_shuffle[i*batch_size : (i+1)*batch_size, :]
                    ])
        
        return train_batches
    
    def get_train_batches_ast_nodes(self):
        return self.__get_train_batches(self.__train_data_ast_nodes)
    
    def get_train_batches_functions_name(self):
        return self.__get_train_batches(self.__train_data_functions_name)
    
    def get_train_batches_variables_name(self):
        return self.__get_train_batches(self.__train_data_variables_name)
    
    def get_train_batches_values(self):
        return self.__get_train_batches(self.__train_data_values)
    
    def get_train_batches_methods(self):
        return [self.get_train_batches_ast_nodes,
                self.get_train_batches_functions_name,
                self.get_train_batches_variables_name,
                self.get_train_batches_values]
    
    def get_valid_batches_ast_nodes(self):
        return self.__valid_batches_ast_nodes
    
    def get_valid_batches_functions_name(self):
        return self.__valid_batches_functions_name
    
    def get_valid_batches_variables_name(self):
        return self.__valid_batches_variables_name
    
    def get_valid_batches_values(self):
        return self.__valid_batches_values
    
    def get_valid_batches_methods(self):
        return [self.get_valid_batches_ast_nodes,
                self.get_valid_batches_functions_name,
                self.get_valid_batches_variables_name,
                self.get_valid_batches_values]
        
    def get_unbalance_class_weights_ast_nodes(self):
        return self.__unbalance_class_weights_ast_nodes
    def get_unbalance_class_weights_functions_name(self):
        return self.__unbalance_class_weights_functions_name
    def get_unbalance_class_weights_variables_name(self):
        return self.__unbalance_class_weights_variables_name
    def get_unbalance_class_weights_values(self):
        return self.__unbalance_class_weights_values
    def get_unbalance_class_weights_methods(self):
        return [self.get_unbalance_class_weights_ast_nodes,
                self.get_unbalance_class_weights_functions_name,
                self.get_unbalance_class_weights_variables_name,
                self.get_unbalance_class_weights_values]
    
    def __read_unbalance_class_weights(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [[i] for i in eval(f.read())]
