# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
import numpy as np
import math
import os
import re
import datetime
import ast
import astunparse

import model
from setting import Path
from setting import tokenize
from data_generator import Generator
from grammar import Grammar

class Predict:
    def __init__(self, paras_list):
        self.__paras_list = paras_list
        self.__paras_base = paras_list[0]
        self.__grammar_checker = Grammar()
        
        self.__prediction_dir = Path.get_prediction_path(self.__paras_base)
        if not os.path.exists(self.__prediction_dir):
            os.makedirs(self.__prediction_dir)
            
        self.__beam_size = self.__paras_base.predict_beam_size
        
        # vocabulary_list = [ast voc, func voc, var voc, value voc]
        # inverse_vocabulary_list = [inv ast voc, inv func voc, inv var voc, inv value voc]
        self.__nl_vocabulary, self.__tree_nodes_vocabulary, self.__vocabulary_list, self.__invert_vocabulary_list = self.__read_vocabulary()
        
        self.__test = self.__paras_base.test
            
            
    ''' predict method '''
    def predict(self):
        self.log = open('predict_log', mode='a')
        self.log.write(str(datetime.datetime.now()) + '\n')
        self.log.write('start predicting\n')
        self.log = open('predict_log', mode='a')
        self.log.write(str(datetime.datetime.now()) + '\n')
        self.log.write('start predicting\n')
        self.log.write('dataset=' + str(self.__paras_base.dataset_path))
        self.log.write(', use_pre_train=' + str(self.__paras_base.use_pre_train))
        self.log.write(', use_semantic=' + str(self.__paras_base.use_semantic_logic_order) + '\n')

        # create graphs and models
        # ast node / function / var / value
        graphs = [tf.Graph(), tf.Graph(), tf.Graph(), tf.Graph()]
        models = []
        for graph, paras in zip(graphs, self.__paras_list):
            with graph.as_default():
                models.append(model.Model(paras))
        model_dir_list = Path.get_model_path_list(self.__paras_base)
        
        # predict 
        with tf.Session(config=self.__gpu_config(), graph=graphs[0]) as sess_0, tf.Session(config=self.__gpu_config(), graph=graphs[1]) as sess_1, tf.Session(config=self.__gpu_config(), graph=graphs[2]) as sess_2, tf.Session(config=self.__gpu_config(), graph=graphs[3]) as sess_3:
            sessions = [sess_0, sess_1, sess_2, sess_3]
            # restore model
            for model_dir, session in zip(model_dir_list, sessions):
                with session.graph.as_default():
                    self.__restore_ckpt(session, model_dir)
            
            descriptions = self.__read_description()
            
            if (self.__test):
                self.log.write('just testing\n')
                self.__predict_one_sentence(descriptions[0], 'test_prediction', sessions, models)
            else:
                for n, description in enumerate(descriptions):
                    write_path = self.__prediction_dir + str(n)
                    self.log.write('num : ' + str(n) + '\n')
                    self.__predict_one_sentence(description, write_path, sessions, models)
        self.log.write('\n')
        self.log.close()
    
    ''' predict one sentence '''
    def __predict_one_sentence(self, description, write_path, sessions, models):
        ast_ = 0
#        func_, var_, value_ = 1, 2, 3
        
        begin_log_probability = -1
        
        # initialize the beam
        begin_unit = Predict.__Beam_Unit(description, ['ast.Module', '{'], begin_log_probability, 1, self.__tree_nodes_vocabulary)
        beam = [begin_unit]
        
        # recoord best result
        max_log_probability_result = Predict.__Beam_Unit(None, None, -1e10, None, None)
                
        self.log.write('recording each result:\n')        
        # predicition
        for i in range(self.__paras_base.max_predict_time):
            ''' predict ''' 
            # separate the beam by different type
            beam_ast = []
            beam_func = []
            beam_var = []
            beam_value = []
            beam_list = [beam_ast, beam_func, beam_var, beam_value]
            
            ''' generate beam data '''
            for beam_unit in beam:
                beam_unit.generate_data(self.__paras_base) 
                # [0|1|2|3+], each beam_unit may have multi predict type
                beam_unit_types = beam_unit.get_predict_type()
                for beam_unit_type in beam_unit_types:
                    beam_list[beam_unit_type].append(beam_unit)
            
            # predict beam
            log_predicted_output = []
            for beam_list_separate, session, nn_model in zip(beam_list, sessions, models):
                log_predicted_output.append(self.__predict_one_beam_with_one_type(session, beam_list_separate, nn_model))
            
            ''' generate new beam '''
            new_beam = []
            for beam_unit_type in range(len(beam_list)):
                for i, beam_unit in enumerate(beam_list[beam_unit_type]):
                    
                    unit_log_predicted_output = log_predicted_output[beam_unit_type][i]
                    unit_traceable_list = beam_unit.traceable_list
                    
                    # all possible layers can append the next predicted node
                    appendable_layers = beam_unit.get_appendable_layers()
                    
                    # if nothing or only '<List>' as parent in the appendable_layers
                    # try to append possibality of <END_Node>
                    # and try to update max_log_probability_result
                    if (False not in [appendable_layer[0] == '<List>' for appendable_layer in appendable_layers]):
                        # this situation, predicting type must be ast node
                        assert beam_unit_type == ast_
                        
                        # the 1st id is the end node
                        log_probability_of_end_node = unit_log_predicted_output[0]
                        # if end the log probability is
                        temp = beam_unit.log_probability * math.pow(beam_unit.predicted_nodes_num, self.__paras_base.short_sentence_penalty_power)
                        temp = temp + log_probability_of_end_node
                        
                        self.log.write('length : ' + str(beam_unit.predicted_nodes_num + 1) + ', log probability before penalty : ' + str(temp) + '\n')
                        self.log.write(str(unit_traceable_list))
                        try:
                            result_code = self.__traceable_list_to_code(unit_traceable_list)
                            self.log.write(result_code)
                        except:
                            self.log.write('unparse error:\n')
                            self.log.write(str(sys.exc_info()))
                        self.log.write('\n')
                        end_log_probability = temp / math.pow(beam_unit.predicted_nodes_num + 1, self.__paras_base.short_sentence_penalty_power)
                        # if better result appears, update the best result
                        if (end_log_probability > max_log_probability_result.log_probability):
                            max_log_probability_result = Predict.__Beam_Unit(description, unit_traceable_list[:], end_log_probability, beam_unit.predicted_nodes_num + 1, self.__tree_nodes_vocabulary)
    
                    # add to new beam
                    for appendable_layer in appendable_layers:
                        # for each appendable layer, there is a unique predict type
                        if not appendable_layer[3] == beam_unit_type:
                            continue
                        
                        for each_id in range(len(unit_log_predicted_output)):
                            
                            # skip '<END_Node>'
                            if beam_unit_type == ast_ and each_id == 0:
                                continue                        
                            
                            each_node = self.__invert_vocabulary_list[beam_unit_type][each_id]
                            each_probability = unit_log_predicted_output[each_id]
                            
                            # penalty of 'unknwon'
                            if (each_id == 0):
                                each_probability -= self.__paras_base.unknwon_log_penalty
                            
                            # do a grammar check
                            if not (self.__grammar_checker.grammar_no_problem(appendable_layer[0], each_node, appendable_layer[2])):
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
                            temp = beam_unit.log_probability * math.pow(beam_unit.predicted_nodes_num, self.__paras_base.short_sentence_penalty_power)
                            temp = temp + each_probability
                            new_probability = temp / math.pow(new_nodes_num, self.__paras_base.short_sentence_penalty_power)
                            
                            new_beam_unit = self.__Beam_Unit(description, new_traceable_list, new_probability, new_nodes_num, self.__tree_nodes_vocabulary)
                            new_beam.append(new_beam_unit)
            
            beam = new_beam
            
            ''' reduce beam max '''
            beam.sort(key=lambda beam_unit:beam_unit.log_probability, reverse=True)
            beam = beam[:self.__beam_size]
            
        self.log.write('\n')
        self.log.write(str(max_log_probability_result.traceable_list) + '\n')            
        ''' generate and write out the code '''
        try:
            # code
            code = self.__traceable_list_to_code(max_log_probability_result.traceable_list)
            if (self.__paras_base.dataset_path == Path.HS_PATH):
                code = self.__hs(code, description)
            # write out
            with open(write_path, 'w', encoding='utf-8') as f:
                f.write(code)
            self.log.write('output : ' + str(code))
        except:
            self.log.write('unparse error:\n')
            self.log.write(str(sys.exc_info()))
        self.log.write('\n\n')
        self.log.flush()

        
    '''
    batch/beam predict for beam search with special model & session
    @beam
    @log_predicted_output
    '''
    def __predict_one_beam_with_one_type(self, session, beam, model):
        if (len(beam) == 0):
            return [[]]
        data_batch = [
                [beam_unit.description for beam_unit in beam],
                [beam_unit.node_list for beam_unit in beam],
                [beam_unit.parent_list for beam_unit in beam],
                [beam_unit.grandparent_list for beam_unit in beam],
                [beam_unit.semantic_units for beam_unit in beam],
                [beam_unit.semantic_unit_children for beam_unit in beam]]        
        
        log_predicted_output = session.run(
                model.log_predicted_output,
                feed_dict={
                        model.input_NL : data_batch[0], # desciption_batch
                        model.input_ast_nodes : data_batch[1], # node_list_batch
                        model.input_ast_parent_nodes : data_batch[2], # parent_list_batch
                        model.input_ast_grandparent_nodes : data_batch[3], # grandparent_list_batch
                        model.input_semantic_units : data_batch[4], # semantic_units_batch
                        model.input_children_of_semantic_units : data_batch[5], # semantic_unit_children_batch
                        model.keep_prob : 1.0})
        
        return log_predicted_output

    
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
                # ast.Set must have a list with at least 1 element, if it is empty, we fill in a stub
                if ((next_node == 'ast.Set') and len(child_list[0]) == 0):
                    new_instance += '[ast.Load()]'
                    
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
    def __restore_ckpt(self, session, model_dir):
        if (os.path.exists(model_dir + 'checkpoint')):
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(model_dir))
            print('restoring model')
        else:
            raise Exception('No Model : ' + model_dir)
    
    ''' read the nl vocabulary & vocabulary list & invert vocabulary list '''
    def __read_vocabulary(self):
        data_dir = self.__paras_list[0].dataset_path
        with open(data_dir + Path.NL_VOCABULARY_PATH, 'r', encoding='utf-8') as f:
            nl_vocabulary = eval(f.read())
        with open(data_dir + Path.TREE_NODES_VOCABULARY_PATH, 'r', encoding='utf-8') as f:
            tree_nodes_vocabulary = eval(f.read())
        vocabulary_list = []
        inv_vocabulary_list = []
        Path_list = [Path.AST_NODES_VOCABULARY_PATH, Path.FUNCTIONS_NAME_VOCABULARY_PATH, Path.VARIABLES_NAME_VOCABULARY_PATH, Path.VALUES_VOCABULARY_PATH]
        for path in Path_list:
            with open(data_dir + path, 'r', encoding='utf-8') as f:
                vocabulary = eval(f.read())
                inv_vocabulary = {v:k for k,v in vocabulary.items()}
                vocabulary_list.append(vocabulary)
                inv_vocabulary_list.append(inv_vocabulary)
        
        return nl_vocabulary, tree_nodes_vocabulary, vocabulary_list, inv_vocabulary_list
    
    '''
    process method with same form with Generator.__Data_provider
    @return descriptions : ids of words numpy forms
    '''
    def __read_description(self):
        descriptions = []
            
        if (self.__paras_base.dataset_path == Path.CONALA_PATH):
            path = self.__paras_base.dataset_path + 'conala-test.json'
            with open(path, 'r', encoding='utf-8') as f:
                null = 'null'
                test_data = eval(f.read())
            
            for data_unit in test_data:
                description_1 = data_unit['intent']
                description_2 = data_unit['rewritten_intent']
                
                description = tokenize(description_1)
                if not (description_2 == 'null') : 
                    description.extend(tokenize(description_2))

                description_ids = self.__get_ids_from_nl_vocabulary(description)
                description_np = np.zeros([self.__paras_base.nl_len])
                for i in range(len(description_ids)):
                    description_np[i] = description_ids[i]
                descriptions.append(description_np)
            return descriptions
        
        if (self.__paras_base.dataset_path == Path.HS_PATH):
            path = self.__paras_base.dataset_path + 'test_hs.in'
            with open(path, 'r', encoding='utf-8') as f:
                test_in = f.readlines()
            
            for description in test_in:
                if (description == ''): continue
                description = tokenize(description)
                description_ids = self.__get_ids_from_nl_vocabulary(description)
                description_np = np.zeros([self.__paras_base.nl_len])
                for i in range(len(description_ids)):
                    description_np[i] = description_ids[i]
                descriptions.append(description_np)
            return descriptions
        
    
    def __get_ids_from_nl_vocabulary(self, words):
        return Generator.get_ids_from_vocabulary(words, self.__nl_vocabulary)    
    
    def __hs(self, code, description):
        d = description.index('NAME_END')
        n = description[:d]
        n = [w.capitalize() for w in n]
        name = ''.join(n)
        code = re.sub('(?<=class )(.+?)(?=\()', name, code)
        code = re.sub('(?<=super\(\).__init__\(\')(.+?)(?=\')', name, code)
        return code
    
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
        
        ''' return [0(ast) | 1(func) | 2(var) | 3(value) +] '''
        def get_predict_type(self):
            return self.__predict_type
        
        ''' return [<parent, depth, position, layer_type>] layer_type=0|1|2|3 same with predict type'''
        def get_appendable_layers(self):
            return self.__appendable_layers
        
        def generate_data(self, paras):
            self.__generate_predict_data(paras)
            self.__generate_appendable_layers()
            self.__check_predict_type()
        
        ''' generate data from traceable list '''
        def __generate_predict_data(self, paras):
            # node_list & parent_list & grandparent_list
            node_list = []
            parent_list = []
            grandparent_list = []
            for i in range(len(self.traceable_list)):
                node = self.traceable_list[i]
                if (node == '{' or node == '}' or node == '<data_point>'): continue
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
            semantic_units, semantic_unit_children = Generator.process_semantic_unit(self.traceable_list, paras)
            semantic_units = semantic_units[:paras.semantic_units_len]
            semantic_unit_children = semantic_unit_children[:paras.semantic_units_len]
            
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
            self.node_list = np.zeros([paras.tree_len])
            self.parent_list = np.zeros([paras.tree_len])
            self.grandparent_list = np.zeros([paras.tree_len])
            self.semantic_units = np.zeros([paras.semantic_units_len])
            self.semantic_unit_children = np.zeros([paras.semantic_units_len, paras.semantic_unit_children_num])   
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
        trace all direct parent layers <parent, depth(relation depth), position(children position), grandparent>
        of next predict node from the traceable list
        for traceable list 'p1, {, c1, c2' , current position is c2, parent=p1, depth=1, position=1, grandparent='<Empty_Node>'
        for traceable list 'p1, {, p2, {, c1, c2' , current position is c2, return [<p2, 1, 1, p1>, <p1, 2, 0, '<Empty_Node>'>]
        for traceable list 'p1, {, p2, {, c1, c2, {' , current position is empty, return [<c2, 1, -1, p2>, <p2, 2, 1, p1>, <p1, 3, 0, '<Empty_Node>'>]
        
        '''        
        def __trace_all_parent_layers(self):
            traceable_list = self.traceable_list
             
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
                grandparent = depth_parent[depth + 1] if (depth + 1) in depth_parent.keys() else '<Empty_Node>'
                all_parent_layers.append([depth_parent[depth], depth, depth_position[depth], grandparent])
            
            return all_parent_layers
        
        '''
        recieve result from __trace_all_parent_layers as input
        reserve all possible layers can append the next predicted node
        '''        
        def __generate_appendable_layers(self):
            all_parent_layers = self.__trace_all_parent_layers()
            
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
            
            self.__appendable_layers = appendable_layers
        
        ''' 
        generate predict type
        and change the appendable_layer[3] to layer type
        '''
        def __check_predict_type(self):
            ast_, func_, var_, value_ = 0, 1, 2, 3
            predict_type = set()
            for appendable_layer in self.__appendable_layers:
                parent = appendable_layer[0]
                position = appendable_layer[2]
                grandparent = appendable_layer[3]
                
                if (parent == 'ast.NameConstant' or parent == 'ast.Str' or parent == 'ast.Num'):
                    layer_type = value_
                elif (parent == 'ast.ClassDef' and position == -1):
                    layer_type = value_
                
                elif (parent == 'ast.FunctionDef' and position == -1):
                    layer_type = func_
                elif (parent == 'ast.arg' and position == -1):
                    layer_type = var_
                elif (parent == 'ast.keyword' and position == -1):
                    layer_type = var_
                elif (parent == 'ast.Name'):
                    if position == -1:
                        if grandparent == 'ast.Call':
                            layer_type = func_
                        elif grandparent == 'ast.Attribute':
                            layer_type = var_
                        else:
                            layer_type = value_  
                    else:
                        layer_type = ast_
                elif (parent == 'ast.Attribute'):
                    if (position == 0):
                        if (grandparent == 'ast.Call'):
                            layer_type = func_
                        else:
                            layer_type = var_
                    else:
                        layer_type = ast_
                elif (parent == '<List>'):
                    layer_type = ast_
                else:
                    layer_type = ast_
                
                predict_type.add(layer_type)
                appendable_layer[3] = layer_type
            
            self.__predict_type = list(predict_type)
        
        def __get_ids_from_tree_nodes_vocabulary(self, nodes):
            return Generator.get_ids_from_vocabulary(nodes, self.__tree_nodes_vocabulary)

if (__name__ == '__main__'):
    tf.reset_default_graph()
    from setting import Parameters
    import sys
    handle = Predict(Parameters.get_paras_list_from_argv(sys.argv))
    handle.predict()
