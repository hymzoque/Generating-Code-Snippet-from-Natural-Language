# -*- coding: utf-8 -*-
"""

"""
class Path:
    DATA_PATH = 'data/'
    HS_PATH = DATA_PATH + 'hearthstone/'
    CONALA_PATH = DATA_PATH + 'conala-corpus/'
    
    GENERATED_PATH = 'generated/'
    TRAIN_DATA_PATH = GENERATED_PATH + 'train_data'
    TEST_DATA_PATH = GENERATED_PATH + 'test_data'
    NL_VOCABULARY_PATH = GENERATED_PATH + 'nl_vocabulary'
    TREE_NODES_VOCABULARY_PATH = GENERATED_PATH + 'tree_nodes_vocabulary'
    AST_NODES_VOCABULARY_PATH = GENERATED_PATH + 'ast_nodes_vocabulary'
    FUNCTIONS_NAME_VOCABULARY_PATH = GENERATED_PATH + 'functions_name_vocabulary'
    VARIABLES_NAME_VOCABULARY_PATH = GENERATED_PATH + 'variables_vocabulary'
    VALUES_VOCABULARY_PATH = GENERATED_PATH + 'values_vocabulary'
    
    UNBALANCE_LOSS_WEIGHT_PATH = GENERATED_PATH + 'unbalance_loss_weight'
    STATISTICS_PATH = GENERATED_PATH + 'statistics'
    
    PRE_TRAIN_DATA_PATH = GENERATED_PATH + 'pre_train_data'
    PRE_TRAIN_WEIGHT_PATH = GENERATED_PATH + 'pre_train_weight/'
    
    __MODEL_PATH = 'model/'
    '''
    '''
    @staticmethod
    def get_model_path_list(paras):
        path = Path.__MODEL_PATH
        path += Path.get_path_scope(paras)
        path += '/'
        return [path + 'ast/', path + 'function/', path + 'variable/', path + 'value']
    
    __PREDICTION_PATH = 'prediction/'
    '''
    '''
    @staticmethod
    def get_prediction_path(paras):
        path = Path.__PREDICTION_PATH
        path += Path.get_path_scope(paras)
        path += '/'
        return path
    
    __SUMMARY_PATH = 'summary/'    
    @staticmethod
    def get_summary_path_list(paras):
        path = Path.__SUMMARY_PATH
        path += Path.get_path_scope(paras)
        path += '/'
        return [path + 'ast/', path + 'function/', path + 'variable/', path + 'value']
    
    @staticmethod
    def get_path_scope(paras):
        n = 'conala_' if (paras.dataset_path == Path.CONALA_PATH) else 'hs_'
        n += 'p_' if (paras.use_pre_train) else 'np_'
        n += 's' if (paras.use_semantic_logic_order) else 'ns'
        return n

class Parameters_base:
    semantic_unit_children_num = 3
    vocabulary_embedding_size = 64
    tree_node_embedding_size = 64
    cnn_deepth = 12
    deep_CNN_kernel_size = [2, 3]
    keep_prob = 0.5
    hidden_layer_width = 768
    
    learning_rate = 1e-4
    weight_decay = 1e-4
        
    train_batch_size = 64
    valid_batch_size = 256
    
    use_pre_train = False
    use_semantic_logic_order = False
    
    test = False

class Parameters_conala_base(Parameters_base):
    nl_len = 78 # max 78
    tree_len = 255 # max 255
    semantic_units_len = 32 # max 32
    unbalance_weight_power = 0.6
    min_vocabulary_count = 4
        
    vocabulary_num = 1191
    tree_node_num = 574
        
    train_times = 300
        
    max_predict_time = 50 # avg 27 max 255 
    predict_beam_size = 20
    unknwon_log_penalty = 0
    short_sentence_penalty = 0.9
        
    dataset_path = Path.CONALA_PATH
        
    
class Parameters_conala_ast_nodes(Parameters_conala_base):
    train_times = 250
    correct_predict_class_num = 83

class Parameters_conala_functions(Parameters_conala_base):
    train_times = 600
    correct_predict_class_num = 188

class Parameters_conala_variables(Parameters_conala_base):
    train_times = 600
    correct_predict_class_num = 116

class Parameters_conala_values(Parameters_conala_base):
    train_times = 600
    correct_predict_class_num = 239
    
    
class Parameters_hs_base(Parameters_base):
    nl_len = 76 # max 76
    tree_len = 723 # max 723
    semantic_units_len = 75 # max 75
    unbalance_weight_power = 0.6
    min_vocabulary_count = 2
        
    vocabulary_num = 462
    tree_node_num = 625
        
    train_times = 150
        
    max_predict_time = 250 # avg 143 max 723
    predict_beam_size = 20
    unknwon_log_penalty = 0
    short_sentence_penalty = 1.1
        
    dataset_path = Path.HS_PATH

class Parameters_hs_ast_nodes(Parameters_hs_base):
    train_times = 150
    correct_predict_class_num = 54

class Parameters_hs_functions(Parameters_hs_base):
    train_times = 600
    correct_predict_class_num = 180

class Parameters_hs_variables(Parameters_hs_base):
    train_times = 600
    correct_predict_class_num = 96

class Parameters_hs_values(Parameters_hs_base):
    train_times = 600
    correct_predict_class_num = 297
    


class Parameters:
    '''
    -c conala dataset (default)
    -h hearthstone dataset
    -p pre_train (default)
    -np no pre_train
    -s semantic_logic_order (default)
    -ns no semantic_logic_order
    '''
    @staticmethod
    def get_paras_list_from_argv(argv):
        # todo
        if ('-h' in argv):
            print('using hearthstone dataset')
            paras_list = [Parameters_conala_ast_nodes(), Parameters_hs_functions(), Parameters_hs_variables(), Parameters_hs_values()]
        elif ('-c' in argv):
            print('using conala dataset')
            paras_list = [Parameters_conala_ast_nodes(), Parameters_conala_functions(), Parameters_conala_variables(), Parameters_conala_values()]
        else:
            print('using conala dataset(default)')
            paras_list = [Parameters_conala_ast_nodes(), Parameters_conala_functions(), Parameters_conala_variables(), Parameters_conala_values()]
        
        if ('-np' in argv):
            print('not using pre train')
            for paras in paras_list:
                paras.use_pre_train = False 
        elif ('-p' in argv):
            print('using pre train')
            for paras in paras_list:
                paras.use_pre_train = True
        else:
            print('not using pre train(default)')
            for paras in paras_list:
                paras.use_pre_train = False
            
        if ('-ns' in argv):
            print('not using semantic logic order')
            for paras in paras_list:
                paras.use_semantic_logic_order = False
        elif ('-s' in argv):
            print('using semantic logic order')
            for paras in paras_list:
                paras.use_semantic_logic_order = True
        else:
            print('not using semantic logic order(default)')
            for paras in paras_list:
                paras.use_semantic_logic_order = False
            
        if ('test' in argv):
            print('test mod')
            for paras in paras_list:
                paras.test = True        
        else:
            for paras in paras_list:
                paras.test = False

        return paras_list

import re
'''
from https://github.com/conala-corpus/conala-baseline/blob/master/eval/conala_eval.py 
tokenize_for_bleu_eval()
'''
def tokenize(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens
  
