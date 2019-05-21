# -*- coding: utf-8 -*-
"""

"""
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

# todo dataset & use pre train -> path 
use_pre_train = False

HS_PATH = 'data/hearthstone'
CONALA_PATH = 'data/conala-corpus/'
dataset_path = CONALA_PATH

