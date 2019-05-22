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
    STATISTICS_PATH = GENERATED_PATH + 'statistics'
    
    MODEL_PATH = 'model/'
    PREDICTION_PATH = 'prediction/'
    

class Parameters:
    @staticmethod
    def get_conala_paras():
        return __Parameters_conala()
    @staticmethod
    def get_hs_paras():
        return __Parameters_hs()
    @staticmethod
    def get_paras_from_argv(argv):
        return
    
    
    
class __Parameters_conala:
    nl_len = 32 # max 32
    tree_len = 256 # max 255
    semantic_units_len = 32 # max 32
    semantic_unit_children_num = 3
    min_vocabulary_count = 5
        
    vocabulary_num = 636
    vocabulary_embedding_size = 128
    tree_node_num = 469
    tree_node_embedding_size = 128
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
        
    use_pre_train = False
    use_semantic_logic_order = True   
     
class __Parameters_hs:
    nl_len = 32 # max 32
    tree_len = 256 # max 255
    semantic_units_len = 32 # max 32
    semantic_unit_children_num = 3
    min_vocabulary_count = 5
        
    vocabulary_num = 636
    vocabulary_embedding_size = 128
    tree_node_num = 469
    tree_node_embedding_size = 128
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
        
    use_pre_train = False
    use_semantic_logic_order = True        