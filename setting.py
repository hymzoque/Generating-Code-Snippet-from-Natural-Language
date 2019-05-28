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
    PRE_TRAIN_DATA_PATH = GENERATED_PATH + 'pre_train_data'
    PRE_TRAIN_WEIGHT_PATH = GENERATED_PATH + 'pre_train_weight.txt'
    STATISTICS_PATH = GENERATED_PATH + 'statistics'
    
    __MODEL_PATH = 'model/'
    '''
    '''
    @staticmethod
    def get_model_path(paras):
        path = Path.__MODEL_PATH
        path += 'conala_' if (paras.dataset_path == Path.CONALA_PATH) else 'hs_'
        path += 'p_' if (paras.use_pre_train) else 'np_'
        path += 's' if (paras.use_semantic_logic_order) else 'ns'
        path += '/'
        return path
    
    __PREDICTION_PATH = 'prediction/'
    '''
    '''
    @staticmethod
    def get_prediction_path(paras):
        path = Path.__PREDICTION_PATH
        path += 'conala_' if (paras.dataset_path == Path.CONALA_PATH) else 'hs_'
        path += 'p_' if (paras.use_pre_train) else 'np_'
        path += 's' if (paras.use_semantic_logic_order) else 'ns'
        path += '/'
        return path
    

class Parameters:
    @staticmethod
    def get_conala_paras():
        return Parameters_conala()
    @staticmethod
    def get_hs_paras():
        return Parameters_hs()
    
    '''
    -c conala dataset (default)
    -h hearthstone dataset
    -p pre_train (default)
    -np no pre_train
    -s semantic_logic_order (default)
    -ns no semantic_logic_order
    '''
    @staticmethod
    def get_paras_from_argv(argv):
        if ('-h' in argv):
            print('using hearthstone dataset')
            paras = Parameters_hs()
        elif ('-c' in argv):
            print('using conala dataset')
            paras = Parameters_conala()
        else:
            print('using conala dataset(default)')
            paras = Parameters_conala()
        
        if ('-np' in argv):
            print('not using pre train')
            paras.use_pre_train = False
        elif ('-p' in argv):
            print('using pre train')
            paras.use_pre_train = True
        else:
            print('not using pre train(default)')
            paras.use_pre_train = False
            
        if ('-ns' in argv):
            print('not using semantic logic order')
            paras.use_semantic_logic_order = False
        elif ('-s' in argv):
            print('using semantic logic order')
            paras.use_semantic_logic_order = True
        else:
            print('not using semantic logic order(default)')
            paras.use_semantic_logic_order = False
        
        return paras
    
    
    
class Parameters_conala:
    nl_len = 32 # max 32
    tree_len = 256 # max 255
    semantic_units_len = 32 # max 32
    semantic_unit_children_num = 3
    min_vocabulary_count = 5
        
    vocabulary_num = 636
    vocabulary_embedding_size = 64
    tree_node_num = 469
    tree_node_embedding_size = 64
    cnn_deepth = 20
    deep_CNN_kernel_size = 2
    keep_prob = 0.5
    hidden_layer_width = 768
        
    train_times = 1000
    learning_rate = 1e-4
    weight_decay = 1e-4
        
    train_batch_size = 64
    valid_batch_size = 256
        
    max_predict_time = 256
    predict_beam_size = 10
    short_sentence_penalty = 0.7
        
    dataset_path = Path.CONALA_PATH
        
    use_pre_train = True
    use_semantic_logic_order = True   
     
class Parameters_hs:
    nl_len = 38 # max 38
    tree_len = 723 # max 723
    semantic_units_len = 75 # max 75
    semantic_unit_children_num = 3
    min_vocabulary_count = 3
        
    vocabulary_num = 311
    vocabulary_embedding_size = 64
    tree_node_num = 391
    tree_node_embedding_size = 64
    cnn_deepth = 20
    deep_CNN_kernel_size = 2
    keep_prob = 0.5
    hidden_layer_width = 768
        
    train_times = 300
    learning_rate = 1e-4
    weight_decay = 1e-4
        
    train_batch_size = 64
    valid_batch_size = 256
        
    max_predict_time = 723
    predict_beam_size = 10
    short_sentence_penalty = 0.7
        
    dataset_path = Path.HS_PATH
        
    use_pre_train = True
    use_semantic_logic_order = True        
