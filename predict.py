# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
import numpy as np
import os
import time
import ast
import astunparse

import model
import setting

'''

'''
class Predict:
    HS_PATH = 'data/hearthstone/'
    CONALA_PATH = 'data/conala-corpus/'
    def __init__(self):
        self.__model_dir = 'model/'
        self.__data_dir = Predict.CONALA_PATH
        self.__prediction_dir = 'prediction/'
        self.__beam_size = setting.predict_beam_size
        self.__read_vocabulary()
    
    ''' test prediction for 1 sentence only '''
    def test_predict(self):
        nn_model = model.Model()
        with tf.Session(config=self.__gpu_config()) as sess:
            # restore model
            self.__restore_ckpt(sess)
            
            start = time.time()
            descriptions = self.__read_description()
            description = descriptions[0]
            write_path = 'test_prediciton/0'
            self.__predict_one_sentence(description, write_path, sess, nn_model)
            print('generate time used : ' + str(time.time() - start))
            
    ''' predict method '''
    def predict(self):
        nn_model = model.Model()
        with tf.Session(config=self.__gpu_config()) as sess:
            # restore model
            self.__restore_ckpt(sess)
            
            descriptions = self.__read_description()
            for n in range(len(descriptions)):
                description = descriptions[n]
                write_path = self.__prediction_dir + str(n)
                self.__predict_one_sentence(description, write_path, sess, nn_model)
        
    
    ''' predict one sentence '''
    def __predict_one_sentence(self, description, write_path, session, model):
        # initialize the beam
        begin_unit = Predict.__Beam_Unit(description, traceable_list=['Module', '{'], 0, self.__tree_nodes_vocabulary)
        begin_unit.generate_data()
        beam = [begin_unit]
        
        # stub
        max_log_probability_result = Predict.__Beam_Unit(None, None, log_probability=-1e10, None)
        
        # predicition
        for i in range(setting.max_predict_time):
            ''' predict ''' 
            data_batch = [
                    [beam_unit.description for beam_unit in beam],
                    [beam_unit.node_list for beam_unit in beam],
                    [beam_unit.parent_list for beam_unit in beam],
                    [beam_unit.grandparent_list for beam_unit in beam],
                    [beam_unit.semantic_units for beam_unit in beam],
                    [beam_unit.semantic_unit_children for beam_unit in beam]]
            # [beam_size(batch_size) x tree_node_num]
            log_predicted_output = self.__predict_one_beam(session, data_batch, model)
            
            ''' generate new beam '''
            new_beam = []
            beam_num = len(beam)
            for i in range(beam_num):
                beam_unit = beam[i]
                unit_log_predicted_output = log_predicted_output[i]
                unit_traceable_list = beam_unit.traceable_list
                
                # [<probability, id>, ...]
                unit_log_predicted_output_with_id = [[unit_log_predicted_output[i], i] for i in range(len(unit_log_predicted_output))]
                unit_log_predicted_output_with_id.sort(key=lambda t : t[0], reverse=True)
                
                # trace all direct parent layers <parent, depth(relation depth), position(current children position)>
                # of next predict node from the traceable list
                # for traceable list 'p1, {, c1, c2' , parent=p1, depth=1, position=2
                # for traceable list 'p1, {, p2, {, c1, c2' , return [<p1, 2, 1>, <p2, 1, 2>]    
                all_parent_layers = self.__trace_all_parent_layers(unit_traceable_list)
                
                # reserve all possible layers can append the next predicted node
                appendable_layers = self.__check_appendable_layers(all_parent_layers)
                
                # if nothing or only '<List>' as parent in the appendable_layers
                # can try to append <END_Node> and append the unit to result
                # and try to update max_log_probability_result
                
                
                
                
                
                # add to new beam
                
                pass
            # '<data_point>' {} -> traceable list
            
            
            
            
            beam = new_beam
            
            ''' reduce beam max '''
            beam.sort(key=lambda beam_unit:beam_unit.log_probability, reverse=True)
            beam = beam[:self.__beam_size]
            
            ''' if max log_probability in beam less than max_result_log_probability, prediction end '''
            if (len(beam) == 0 or beam[0].log_probability < max_log_probability_result.log_probability):
                break
            ''' generate new beam data '''
            for beam_unit in beam:
                beam_unit.generate_data()
            
        ''' generate and write out the code '''
        # construct the ast
#        max_log_probability_result
        
        return

    
    '''
    batch/beam predict for beam search
    @data_batch
    @log_predicted_output
    '''
    def __predict_one_beam(self, session, data_batch, model):
        if (len(data_batch) != 6): raise Exception('need data_batch length = 6')
        log_predicted_output = session.run(
                [model.log_predicted_output],
                feed_dict={
                        model.input_NL : data_batch[0], # desciption_batch
                        model.input_ast_nodes : data_batch[1], # node_list_batch
                        model.input_ast_parent_nodes : data_batch[2], # parent_list_batch
                        model.input_ast_grandparent_nods : data_batch[3], # grandparent_list_batch
                        model.input_semantic_units : data_batch[4], # semantic_units_batch
                        model.input_children_of_semantic_units : data_batch[5], # semantic_unit_children_batch
                        model.keep_prob : 1.0                       
                        })
        
        return log_predicted_output
    
    '''
    trace all direct parent layers <parent, depth(relation depth), position(children position)>
    of next predict node from the traceable list
    for traceable list 'p1, {, c1, c2' , parent=p1, depth=1, position=1
    for traceable list 'p1, {, p2, {, c1, c2' , return [<p1, 2, 0>, <p2, 1, 1>]    
    '''
    def __trace_all_parent_layers(self, traceable_list):
        # for each depth only have 1 parent
        depth_position = {}
        current_depth = 1
        for i in reversed(range(len(traceable_list))):
            next_node = traceable_list[i]
            # skip the data point mark
            if (next_node == '<data_point>'): 
                continue
            if (next_node == '{'):
                continue
            if (next_node == '}'):
                continue
            # 
            
        
        return
    '''
    recieve result from __trace_all_parent_layers as input
    reserve all possible layers can append the next predicted node
    '''
    def __check_appendable_layers(self, all_parent_layers):
        return
    

    
    ''' gpu config '''            
    def __gpu_config(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return config
    
    ''' restore the model '''    
    def __restore_ckpt(self, session):
        dir_path = self.__model_dir
        if (os.path.exists(dir_path + 'checkpoint')):
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(dir_path))
            print('restoring model')
        else:
            raise Exception('No Model')
    
    ''' read the nl & tree node vocabulary '''
    def __read_vocabulary(self):
        with open(self.__data_dir + 'nl_vocabulary', 'r') as f:
            self.__nl_vocabulary = eval(f.read())
        with open(self.__data_dir + 'tree_nodes_vocabulary', 'r') as f:
            self.__tree_nodes_vocabulary = eval(f.read())
    
    ''' '''
    def __read_description(self):
        path = self.__data_dir + 'test_data'
        with open(path, 'r') as f:
            test_data = eval(f.read())
        # todo
#        np.zeros
        
        
        return # ids of words np forms
    
    
    
    '''
    Beam unit receive a traceable nodes list(which is same defined by class Generator)
    and generate data to fit the nn model from the traceable nodes list
    '''
    class __Beam_Unit:
        def __init__(self, description, traceable_list, log_probability, tree_nodes_vocabulary):
            self.description = description # ids
            self.traceable_list = traceable_list
            self.log_probability = log_probability
            self.__tree_nodes_vocabulary = tree_nodes_vocabulary
        
        ''' generate data from traceable list '''
        def generate_data(self):
            # node_list & parent_list & grandparent_list
            node_list = []
            parent_list = []
            grandparent_list = []
            for i in range(len(self.traceable_list)):
                node = self.traceable_list[i]
                if (node == '{' or node == '}'): continue
                # for each node
                node_list.append(node)
                # find parent and grandparent from traceable list
                parent_found = False
                grandparent_found = False
                depth = 0
                for j in reversed(range(i)):
                    next_node = self.traceable_list[j]
                    if (next_node == '{'):
                        depth += 1
                        continue
                    if (next_node == '}'):
                        depth -= 1
                        continue
                    if (next_node == '<data_point>'):
                        continue
                    if ((not parent_found) and depth == 1):
                        parent_list.append(next_node)
                        parent_found = True
                        continue
                    if ((not grandparent_found) and depth == 2):
                        grandparent_list.append(next_node)
                        grandparent_found = True
                        break
                if not (parent_found): parent_list.append('<Empty_Node>')
                if not (grandparent_found): grandparent_list.append('<Empty_Node>')
            
            # semantic_units & semantic_unit_children
            semantic_units, semantic_unit_children = self.__process_semantic_unit(self.traceable_list)
            
            # get ids
            node_list = self.__get_ids_from_tree_nodes_vocabulary(node_list)
            parent_list = self.__get_ids_from_tree_nodes_vocabulary(parent_list)
            grandparent_list = self.__get_ids_from_tree_nodes_vocabulary(grandparent_list)
            semantic_units = self.__get_ids_from_tree_nodes_vocabulary(semantic_units)
            semantic_unit_children_ids = []
            for children in semantic_unit_children:
                semantic_unit_children_ids.append(self.__get_ids_from_tree_nodes_vocabulary(children))
            semantic_unit_children = semantic_unit_children_ids
            
            # fill 0 in spare place(using numpy)
            self.node_list = np.zeros([setting.Tree_len])
            self.parent_list = np.zeros([setting.Tree_len])
            self.grandparent_list = np.zeros([setting.Tree_len])
            self.semantic_units = np.zeros([setting.Semantic_Units_len])
            self.semantic_unit_children = np.zeros([setting.Semantic_Units_len, setting.Semantic_Unit_children_num])   
            for j in range(len(node_list)):
                self.node_list[j] = node_list[j]
            for j in range(len(parent_list)):
                self.parent_list[j] = parent_list[j]
            for j in range(len(grandparent_list)):
                self.grandparent_list[j] = grandparent_list[j]
            for j in range(len(semantic_units)):
                self.semantic_units[j] = semantic_units[j]
            if (len(semantic_unit_children) > 0):
                for j in range(len(semantic_unit_children)):
                    for k in range(len(semantic_unit_children[j])):
                        self.semantic_unit_children[j][k] = semantic_unit_children[j][k]
            
        '''
        get semantic_unit_list and semantic_unit_children_list by processing traceable_node_list
        same implement with __process_semantic_unit in class Generator
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
        
        ''' 
        @return : whether the node is a semantic unit 
        same implement with __is_semantic_node in class Generator
        '''
        def __is_semantic_node(self, node):
            return (node == 'ast.Call' or
                    node == 'ast.Attribute' or
                    node == 'ast.Assign' or
                    node == 'ast.AugAssign' or
                    node == 'ast.While' or
                    node == 'ast.If')
        ''' 
        score the child's contribution of semantic information
        same implement with __semantic_child_score in class Generator
        '''
        def __semantic_child_score(self, is_data_point, depth):
            reward = 2.5 if is_data_point else 0.0
            return reward - depth
        '''
        get id from tree nodes vocabulary, if not in vocabulary return 'unknown':0
        same implement with __get_ids_from_tree_nodes_vocabulary in class Generator
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


handle = Predict()
handle.predict()