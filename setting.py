# -*- coding: utf-8 -*-
"""

"""
NL_len = 32 # max 32
Tree_len = 256 # max 255
Semantic_Units_len = 32 # max 32
Semantic_Unit_children_num = 3

use_pre_train = False
vocabulary_num = 636
vocabulary_embedding_size = 128
tree_node_num = 468
tree_node_embedding_size = 128
cnn_deepth = 20
deep_CNN_kernel_size = 2
keep_prob = 0.5
hidden_layer_width = 768

train_times = 2000
learning_rate = 1e-4
weight_decay = 1e-4

train_batch_size = 64
valid_batch_size = 256

max_predict_time = 256
predict_beam_size = 2