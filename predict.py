# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
import numpy as np
import math
import os
import time
import ast
import astunparse

import model
import setting
from data_generator import Generator

class Predict:
    def __init__(self):
        self.__model_dir = 'model/'
        self.__data_dir = setting.CONALA_PATH
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
        begin_unit = Predict.__Beam_Unit(description, traceable_list=['Module', '{'], 0, 1, self.__tree_nodes_vocabulary)
        begin_unit.generate_data()
        beam = [begin_unit]
        
        # recoord best result
        max_log_probability_result = Predict.__Beam_Unit(None, None, -1e10, None)
        
        # predicition
        for i in range(setting.max_predict_time):
            ''' predict ''' 
            # [beam_size(batch_size) x tree_node_num]
            log_predicted_output = self.__predict_one_beam(session, beam, model)
            
            ''' generate new beam '''
            new_beam = []
            beam_num = len(beam)
            for i in range(beam_num):
                beam_unit = beam[i]
                unit_log_predicted_output = log_predicted_output[i]
                unit_traceable_list = beam_unit.traceable_list
                
                # trace all direct parent layers <parent, depth(relation depth), position(children position)>
                # of next predict node from the traceable list
                # for traceable list 'p1, {, c1, c2' , current position is c2, parent=p1, depth=1, position=1
                # for traceable list 'p1, {, p2, {, c1, c2' , current position is c2, return [<p2, 1, 1>, <p1, 2, 0>]
                # for traceable list 'p1, {, p2, {, c1, c2, {' , current position is empty, return [<c2, 1, -1>, <p2, 2, 1>, <p1, 3, 0>] 
                all_parent_layers = self.__trace_all_parent_layers(unit_traceable_list)
                
                # reserve all possible layers that can append the next predicted node
                appendable_layers = self.__check_appendable_layers(all_parent_layers)
                
                # if nothing or only '<List>' as parent in the appendable_layers
                # try to append possibality of <END_Node>
                # and try to update max_log_probability_result
                if (False not in [appendable_layer[0] == '<List>' for appendable_layer in appendable_layers]):
                    # the 2nd id is the end node
                    log_probability_of_end_node = unit_log_predicted_output[1]
                    # if end the log probability is
                    temp = beam_unit.log_probability * math.pow(beam_unit.predicted_nodes_num, setting.short_sentence_penalty)
                    twmp = temp + log_probability_of_end_node
                    end_log_probability = temp / math.pow(beam_unit.predicted_nodes_num + 1, setting.short_sentence_penalty)
                    # if better result appears, update the best result
                    if (end_log_probability > max_log_probability_result.log_probability):
                        max_log_probability_result = Predict.__Beam_Unit(description, unit_traceable_list[:], end_log_probability, beam_unit.predicted_nodes_num + 1, self.__tree_nodes_vocabulary)
                    # todo predict log
                
                # add to new beam
                # for each appendable layer, 
                for appendable_layer in appendable_layers:
                    for each_id in range(len(unit_log_predicted_output)):
                        # skip '<END_Node>'
                        if (each_id == 1):
                            continue                        
                        
                        each_node = self.__invert_tree_nodes_vocabulary(each_id)
                        each_probability = unit_log_predicted_output[each_id]
                        # do a grammar check
                        if not (self.__grammar_no_problem(appendable_layer[0], each_node, appendable_layer[2])):
                            continue
                        # new traceable list
                        new_traceable_list = unit_traceable_list[:]
                        layer_depth = appendable_layer[1]
                        for t in range(layer_depth - 1):
                            new_traceable_list.append('}')
                        # <list> or ast node
                        if (each_node == '<List>' or 'ast.' in each_node):
                            new_traceable_list.append(each_node)
                            new_traceable_list.append('{')
                        # string/<data_point>
                        else: 
                            new_traceable_list.append('<data_point>')
                            new_traceable_list.append(each_node)
                        # new nodes num
                        new_nodes_num = beam_unit.predicted_nodes_num + 1
                        # new probability
                        temp = beam_unit.log_probability * math.pow(beam_unit.predicted_nodes_num, setting.short_sentence_penalty)
                        temp = temp + each_probability
                        new_probability = temp / math.pow(new_nodes_num, setting.short_sentence_penalty)
                        
                        new_beam_unit = self.__Beam_Unit(description, new_traceable_list, new_probability, new_nodes_num, self.__tree_nodes_vocabulary)
                        new_beam.append(new_beam_unit)
            
            beam = new_beam
            
            ''' reduce beam max '''
            beam.sort(key=lambda beam_unit:beam_unit.log_probability, reverse=True)
            beam = beam[:self.__beam_size]
            
            ''' generate new beam data '''
            for beam_unit in beam:
                beam_unit.generate_data()
            
        ''' generate and write out the code '''
        # code
        code = self.__traceable_list_to_code(max_log_probability_result.traceable_list)
        # write out
        with open(write_path, 'w') as f:
            f.write(code)
        
    '''
    batch/beam predict for beam search
    @beam
    @log_predicted_output
    '''
    def __predict_one_beam(self, session, beam, model):
        data_batch = [
                [beam_unit.description for beam_unit in beam],
                [beam_unit.node_list for beam_unit in beam],
                [beam_unit.parent_list for beam_unit in beam],
                [beam_unit.grandparent_list for beam_unit in beam],
                [beam_unit.semantic_units for beam_unit in beam],
                [beam_unit.semantic_unit_children for beam_unit in beam]]        
        
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
    for traceable list 'p1, {, c1, c2' , current position is c2, parent=p1, depth=1, position=1
    for traceable list 'p1, {, p2, {, c1, c2' , current position is c2, return [<p2, 1, 1>, <p1, 2, 0>]
    for traceable list 'p1, {, p2, {, c1, c2, {' , current position is empty, return [<c2, 1, -1>, <p2, 2, 1>, <p1, 3, 0>]
    '''
    def __trace_all_parent_layers(self, traceable_list):            
        depth_position = {1:-1}
        depth_parent = {}
        current_depth = 0
        max_depth = current_depth
        for i in reversed(range(len(traceable_list))):
            next_node = traceable_list[i]
            # skip the data point mark
            if (next_node == '<data_point>'): 
                continue
            # 
            if (next_node == '{'):
                current_depth += 1
                # update max depth
                if (current_depth > max_depth):
                    max_depth = current_depth
                continue
            # 
            if (next_node == '}'):
                current_depth -= 1
                continue
            
            # once you have reached the max depth, then you won't need care the node in (depth < max depth)
            if (current_depth < max_depth):
                continue
            
            # plus position
            if ((current_depth + 1) in depth_position.keys()):
                depth_position[current_depth + 1] += 1
            # or register position
            else:
                depth_position[current_depth + 1] = 0
            
            # register parent
            if (current_depth > 0 and current_depth not in depth_parent.keys()):
                depth_parent[current_depth] = next_node
            
        all_parent_layers = []
        for depth in depth_parent.keys():
            all_parent_layers.append([depth_parent[depth], depth, depth_position[depth]])
        
        return all_parent_layers
    '''
    recieve result from __trace_all_parent_layers as input
    reserve all possible layers can append the next predicted node
    '''
    def __check_appendable_layers(self, all_parent_layers):
        appendable_layers = []
        for parent_layer in all_parent_layers:
            # <List> is appendable
            if (parent_layer[0] == '<List>'):
                appendable_layers.append(parent_layer)
                continue
            # ast node with not enough children is appendable
            # and if there is a appendable ast node, we only need this layer, deeper layer will be dropped
            elif ('ast.' in parent_layer[0]):
                node_name = parent_layer[0][4:]
                attr_num = len(getattr(ast, node_name)._fields)
                position = parent_layer[2]
                # appendable
                if (position + 1 < attr_num):
                    appendable_layers.append(parent_layer)
                    break
                else:
                    continue
                
            # string should not be a parent
            else:
                raise Exception('string as parent should not be a appendable layer, ' + str(parent_layer))
        
        return appendable_layers
    
    '''
    check grammar by parent, child and position of last child(position of this child - 1)
    may need to extend
    '''
    def __grammar_no_problem(self, parent, child, position):
        # <List> can not have child <List>
        if (parent == '<List>' and child == '<List>'):
            return False
        # <List> can not have string as child(not strict)
        
        # Name and Str must have string as first child(not strict)
        
        # ClassDef or FunctionDef must have string as first child(name)
        if ((parent == 'ast.ClassDef' or parent == 'ast.FunctionDef') and position == -1):
            if (child == '<List>' or child == '<Empty_List>' or child == '<Empty_Node>' or 'ast.' in child):
                return False
        return True
               
    
    '''
    traceable list -> AST -> code
    '''
    def __traceable_list_to_code(self, traceable_list):
        # delete '}' in the tail of list
        for i in reversed(range(len(traceable_list))):
            if (traceable_list[i] == '}'):
                continue
            else:
                break
        traceable_list = traceable_list[:i+1]
        
        # each stack is a child list
        stacks = [[]]
        for i in reversed(range(len(traceable_list))):
            next_node = traceable_list[i]
            if (next_node == '<data_point>'): # skip
                continue
            elif (next_node == '}'): # create a new stack(child list)
                stacks.append([])
                continue
            elif (next_node == '<Empty_List>'): # append a [] as a child
                stacks[len(stacks) - 1].append([])
                continue
            elif (next_node == '<None_Node>'): # append a None as a child
                stacks[len(stacks) - 1].append(None)
                continue
            elif (next_node == '<List>'): # create a list use the top stack, and append the list as a child
                l = stacks.pop()
                l.reverse()
                if (len(stacks) == 0):
                    stacks.append([l])
                else:
                    stacks[len(stacks) - 1].append(l)
                continue
            elif ('ast.' in next_node): # ast node
                child_list = stacks.pop()
                # eval('ast.XXX(..., child_list[1], child_list[0])')
                new_instance = next_node + '('
                for i in reversed(range(len(child_list))):
                    new_instance += 'child_list[' + str(i) + ']'
                    if (i != 0):
                        new_instance += ', '
                new_instance += ')'
                new_node = eval(new_instance)
                if (len(stacks) == 0):
                    stacks.append([new_node])
                else:
                    stacks[len(stacks) - 1].append(new_node)
                continue
            elif (next_node == '{'): # skip, '{' must followed by a <List> or a ast node
                continue
            else: # data
                # data type
                # bool
                if (next_node == 'True'): 
                    next_node = True
                elif (next_node == 'False'): 
                    next_node = False
                else:
                    # int
                    try:
                        next_node = int(next_node)
                    except:
                        # float
                        try:
                            next_node = float(next_node)
                        # str
                        except:
                            next_node = str(next_node)
                stacks[len(stacks) - 1].append(next_node)
                continue
        root = stacks[0][0]
        code = astunparse.unparse(root)
        return code
    
    
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
    
    ''' read the nl vocabulary & tree nodes vocabulary & invert tree nodes vocabulary'''
    def __read_vocabulary(self):
        with open(self.__data_dir + 'nl_vocabulary', 'r') as f:
            self.__nl_vocabulary = eval(f.read())
        with open(self.__data_dir + 'tree_nodes_vocabulary', 'r') as f:
            self.__tree_nodes_vocabulary = eval(f.read())
        self.__invert_tree_nodes_vocabulary = {v:k for k,v in self.__tree_nodes_vocabulary.items()}
    '''
    process method with same form with Generator.__Data_provider
    @return descriptions : ids of words numpy forms
    '''
    def __read_description(self):
        # todo multi dataset
        path = self.__data_dir + 'conala-test.json'
        with open(path, 'r') as f:
            null = 'null'
            test_data = eval(f.read())
        
        descriptions = []
        for data_unit in test_data:
            description = data_unit['rewritten_intent']
            if (description == 'null') : continue
            description = re.sub('[\'\"`]', '', description).strip()
            description = description.split(' ')
            description_ids = self.__get_ids_from_nl_vocabulary(description)
            description_np = np.zeros([setting.nl_len])
            for i in range(len(description_ids)):
                description_np[i] = description_ids[i]
            descriptions.append(description_np)
        return descriptions
    
    def __get_ids_from_nl_vocabulary(self, words):
        return Generator.get_ids_from_nl_vocabulary(words, self.__nl_vocabulary)    
    
    '''
    Beam unit receive a traceable nodes list(which is same defined by class Generator)
    and can generate data to fit the nn model from the traceable nodes list
    include : node_list/parent_list/grandparent_list/semantic_units/semantic_unit_children
    '''
    class __Beam_Unit:
        def __init__(self, description, traceable_list, log_probability, predicted_nodes_num, tree_nodes_vocabulary):
            self.description = description # ids
            self.traceable_list = traceable_list # str
            self.log_probability = log_probability
            self.predicted_nodes_num = predicted_nodes_num
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
            semantic_units, semantic_unit_children = Generator.process_semantic_unit(self.traceable_list)
            
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
            self.node_list = np.zeros([setting.tree_len])
            self.parent_list = np.zeros([setting.tree_len])
            self.grandparent_list = np.zeros([setting.tree_len])
            self.semantic_units = np.zeros([setting.semantic_units_len])
            self.semantic_unit_children = np.zeros([setting.semantic_units_len, setting.semantic_unit_children_num])   
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


        def __get_ids_from_tree_nodes_vocabulary(self, nodes):
            return Generator.get_ids_from_tree_nodes_vocabulary(nodes, self.__tree_nodes_vocabulary)


handle = Predict()
#handle.test_predict()
handle.predict()