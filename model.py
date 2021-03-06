# -*- coding: utf-8 -*-
"""


"""
import tensorflow as tf

class Model:
    def __init__(self, paras):
        self.__paras = paras
        self.__place_alloc()
        self.__bulid_graph()    
        
    '''
    
    '''    
    def __place_alloc(self):
        # None=batch_size
        # descriptions 
        self.input_NL = tf.placeholder(tf.int32, shape=[None, self.__paras.nl_len])
        # nodes of predicted ast
        self.input_ast_nodes = tf.placeholder(tf.int32, shape=[None, self.__paras.tree_len])
        self.input_ast_parent_nodes = tf.placeholder(tf.int32, shape=[None, self.__paras.tree_len])
        self.input_ast_grandparent_nodes = tf.placeholder(tf.int32, shape=[None, self.__paras.tree_len])
        # nodes of predicted semantic units
        self.input_semantic_units = tf.placeholder(tf.int32, shape=[None, self.__paras.semantic_units_len])
        self.input_children_of_semantic_units = tf.placeholder(tf.int32, shape=[None, self.__paras.semantic_units_len, self.__paras.semantic_unit_children_num])
        # indexes of correct output
        self.correct_output = tf.placeholder(tf.float32, shape=[None, self.__paras.correct_predict_class_num])
#        # unbalance weights table
        self.unbalance_class_weights = tf.placeholder(tf.float32, shape=[self.__paras.correct_predict_class_num, 1])
        # keep_prob = 1 - dropout
        self.keep_prob = tf.placeholder(tf.float32)
        
        # use default glorot_uniform_initializer
        self.vocabulary_embedding = tf.get_variable('vocabulary_embedding', shape=[self.__paras.vocabulary_num, self.__paras.vocabulary_embedding_size])
        self.tree_node_embedding = tf.get_variable('tree_node_embedding', shape=[self.__paras.tree_node_num, self.__paras.tree_node_embedding_size])
        
        # pre train embedding
        if (self.__paras.use_pre_train):
            self.pre_train_tree_node_embedding = tf.get_variable('pre_train_tree_node_embedding', shape=[self.__paras.tree_node_num, self.__paras.tree_node_embedding_size], trainable=True)

    '''
    
    '''
    def __bulid_graph(self):
        '''
        input of description
        '''
        nl_embedding = tf.nn.embedding_lookup(self.vocabulary_embedding, self.input_NL)
        nl_features = self.__deep_CNN(nl_embedding, self.__paras.vocabulary_embedding_size)
        
        ''' 
        input of parent relation tree nodes
        '''
        # [batch size, tree length, embedding size]
        ast_nodes_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_ast_nodes)
        ast_parent_nodes_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_ast_parent_nodes)
        ast_grandparent_nodes_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_ast_grandparent_nodes)
        # [batch size, tree length, 3, embedding size]  (batch, height, width, channels) -- 'channel last'
        parent_stack = tf.stack([ast_nodes_embedding, ast_parent_nodes_embedding, ast_grandparent_nodes_embedding], 2)
        # [batch size, tree length, 1, embedding size]
        temp = tf.layers.conv2d(parent_stack, self.__paras.tree_node_embedding_size, [1, 3])
        # [batch size, tree length, embedding size]
        temp = tf.reduce_max(temp, axis=2)
        temp = tf.nn.relu(temp)
        
        # [batch size, tree length, embedding size]
        tree_parent_features = self.__deep_CNN(temp, self.__paras.tree_node_embedding_size)
        
        # pre train and 2 channel
        if (self.__paras.use_pre_train):
            ast_nodes_embedding_2 = tf.nn.embedding_lookup(self.pre_train_tree_node_embedding, self.input_ast_nodes)
            ast_parent_nodes_embedding_2 = tf.nn.embedding_lookup(self.pre_train_tree_node_embedding, self.input_ast_parent_nodes)
            ast_grandparent_nodes_embedding_2 = tf.nn.embedding_lookup(self.pre_train_tree_node_embedding, self.input_ast_grandparent_nodes)
            parent_stack_2 = tf.stack([ast_nodes_embedding_2, ast_parent_nodes_embedding_2, ast_grandparent_nodes_embedding_2], 2)
            temp_2 = tf.layers.conv2d(parent_stack_2, self.__paras.tree_node_embedding_size, [1, 3])
            temp_2 = tf.reduce_max(temp_2, axis=2)
            temp_2 = tf.nn.relu(temp_2)        
            tree_parent_features_2 = self.__deep_CNN(temp_2, self.__paras.tree_node_embedding_size)        
            # max pooling 
            feature_stack = tf.stack([tree_parent_features, tree_parent_features_2], axis=1)
            tree_parent_features = self.__multi_channel_pooling(feature_stack)
        
        ''' 
        input of semantic order tree nodes 
        '''
        if (self.__paras.use_semantic_logic_order):
            semantic_units_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_semantic_units)
            semantic_em_stack = tf.expand_dims(semantic_units_embedding, axis=2)
            # loop
            for i in range(self.__paras.semantic_unit_children_num):
                child = self.input_children_of_semantic_units[:, :, i]
                child_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, child)
                child_embedding = tf.expand_dims(child_embedding, axis=2)
                semantic_em_stack = tf.concat([semantic_em_stack, child_embedding], axis=2)            
            
            temp = tf.layers.conv2d(semantic_em_stack, self.__paras.tree_node_embedding_size, [1, self.__paras.semantic_unit_children_num + 1])
            temp = tf.reduce_max(temp, axis=2)
            temp = tf.nn.relu(temp) 
            
            semantic_order_features = self.__deep_CNN(temp, self.__paras.tree_node_embedding_size)
            
            # pre train and 2 channel
            if (self.__paras.use_pre_train):
                semantic_units_embedding_2 = tf.nn.embedding_lookup(self.pre_train_tree_node_embedding, self.input_semantic_units)
                semantic_em_stack_2 = tf.expand_dims(semantic_units_embedding_2, axis=2)
                for i in range(self.__paras.semantic_unit_children_num):
                    child = self.input_children_of_semantic_units[:, :, i]
                    child_embedding_2 = tf.nn.embedding_lookup(self.pre_train_tree_node_embedding, child)
                    child_embedding_2 = tf.expand_dims(child_embedding_2, axis=2)
                    semantic_em_stack_2 = tf.concat([semantic_em_stack_2, child_embedding_2], axis=2)            
                
                temp_2 = tf.layers.conv2d(semantic_em_stack_2, self.__paras.tree_node_embedding_size, [1, self.__paras.semantic_unit_children_num + 1])
                temp_2 = tf.reduce_max(temp_2, axis=2)
                temp_2 = tf.nn.relu(temp_2) 
                semantic_order_features_2 = self.__deep_CNN(temp_2, self.__paras.tree_node_embedding_size)                
                # max pooling
                feature_stack = tf.stack([semantic_order_features, semantic_order_features_2], axis=1)
                semantic_order_features = self.__multi_channel_pooling(feature_stack)
            
        '''
        attention
        
        '''
        # pooling
        # [batch size, nl/tree embedding size]
        nl_features_pooling = self.__max_feature_pooling(nl_features)
        tree_parent_features_pooling = self.__max_feature_pooling(tree_parent_features)
        if (self.__paras.use_semantic_logic_order):
            semantic_order_features_pooling = self.__max_feature_pooling(semantic_order_features)
        
        # attention
        attention_result_nl = self.__attention(nl_features, tree_parent_features_pooling)
        if (self.__paras.use_semantic_logic_order):
            attention_result_nl_2 = self.__attention(nl_features, semantic_order_features_pooling)
        attention_result_tree_parent = self.__attention(tree_parent_features, nl_features_pooling)
        if (self.__paras.use_semantic_logic_order):
            attention_result_semantic_order = self.__attention(semantic_order_features, nl_features_pooling)
        
        # concat the features
        temp = [nl_features_pooling,
                tree_parent_features_pooling,
                attention_result_nl,
                attention_result_tree_parent]
        if (self.__paras.use_semantic_logic_order):
            temp.append(semantic_order_features_pooling)
            temp.append(attention_result_nl_2)
            temp.append(attention_result_semantic_order)
        full_connect_input = tf.concat(temp, axis = 1)
        
        ''' 
        full connect with single hidden layer
        '''
        w1 = tf.get_variable('hidden_fc_w1', shape=[int(full_connect_input.shape[1]), self.__paras.hidden_layer_width], initializer=self.__initializer())
        b1 = tf.get_variable('hidden_fc_b1', shape=[self.__paras.hidden_layer_width], initializer=self.__initializer())
        w2 = tf.get_variable('hidden_fc_w2', shape=[self.__paras.hidden_layer_width, self.__paras.correct_predict_class_num], initializer=self.__initializer())
        b2 = tf.get_variable('hidden_fc_b2', shape=[self.__paras.correct_predict_class_num], initializer=self.__initializer())
        temp = tf.nn.tanh(tf.matmul(full_connect_input, w1) + b1)
        temp = tf.nn.dropout(temp, self.keep_prob)
        # logits
        logits = tf.matmul(temp, w2) + b2
        # softmax output   [batch size, output class num]
        self.predicted_output = tf.nn.softmax(logits)
        # log of output, for the precision of probability product
        self.log_predicted_output = tf.log(tf.clip_by_value(self.predicted_output, 1e-100, 1.0))
        ''' 
        output and optimize
        '''        
#        # unbalance weights 
#        # [batch size]
        unbalance_sample_weights = tf.reduce_max(tf.matmul(self.correct_output, self.unbalance_class_weights), axis=1)
        
        # [batch size]
        predict_result = tf.equal(tf.argmax(self.predicted_output, 1), tf.argmax(self.correct_output, 1))
        predict_result = tf.cast(predict_result, tf.float32)
        self.accuracy = tf.reduce_mean(predict_result)
        
#        batch_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.correct_output, logits=logits)
        batch_cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.correct_output, logits=logits, weights=unbalance_sample_weights)
        # weighted loss
        self.cross_entropy = tf.reduce_mean(batch_cross_entropy)
        self.optimize = tf.contrib.opt.AdamWOptimizer(weight_decay=self.__paras.weight_decay, learning_rate=self.__paras.learning_rate).minimize(self.cross_entropy)
        
#        tf.summary.scalar('weighted_loss', self.cross_entropy)
        tf.summary.scalar('loss', self.cross_entropy)
        tf.summary.scalar('acc', self.accuracy)
        self.merged = tf.summary.merge_all()
        
    ''' weight initializer '''
    def __initializer(self):
        return tf.truncated_normal_initializer(stddev=0.1)        
    
    '''
    resnet deep CNN
    '''    
    def __deep_CNN(self, tensor, channel_size):
        loop_time = int(self.__paras.cnn_deepth / 2)
        stack = []
        for kernel_size in self.__paras.deep_CNN_kernel_size:
            temp = tensor
            for i in range(loop_time):
                temp = tf.layers.conv1d(temp, channel_size , kernel_size, padding='same')
                temp = tf.nn.relu(temp)
                temp = tf.layers.conv1d(temp, channel_size , kernel_size, padding='same')
                temp = tf.add_n([temp, tensor])
                temp = tf.nn.relu(temp)
            stack.append(temp)
        stack = tf.stack(stack, axis=1)
        return self.__multi_channel_pooling(stack)
    
    def __deep_CNN_1_channel(self, tensor, channel_size, kernel_size):
        loop_time = int(self.__paras.cnn_deepth / 2)
        for i in range(loop_time):
            temp = tf.layers.conv1d(tensor, channel_size , kernel_size, padding='same')
            temp = tf.nn.relu(temp)
            temp = tf.layers.conv1d(temp, channel_size , kernel_size, padding='same')
            temp = tf.add_n([temp, tensor])
            tensor = tf.nn.relu(temp)
        return tensor
    
    
    '''
    @tensor : [batch_size, tree/nl_length(height), tree/nl_embedding_size(width)]
    @return : [batch_size, tree/nl_embedding_size]
    '''
    def __max_feature_pooling(self, tensor):
        height = int(tensor.get_shape()[1])
        width = int(tensor.get_shape()[2])
        tensor_expand = tf.expand_dims(tensor, -1)
        # [batch_size, 1, embedding size, 1]
        result = tf.nn.max_pool(tensor_expand, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        return tf.reshape(result, [-1, width])

    '''
    calculate attention for each set of features
    @features : [batch_size, length, embedding_size_1]
    @controlling_vector : [batch_size, embedding_size_2]
    @return : [batch_size, embedding_size_1]
    '''    
    def __attention(self, features, controlling_vector):
        # [embedding size 1, embedding size 2]
        initial_value = tf.truncated_normal(shape=[int(features.get_shape()[2]), int(controlling_vector.get_shape()[1])], stddev=0.1)
        attention_matrix = tf.Variable(initial_value)
        # [batch size, length, embeding size 2]
        temp = tf.einsum("ijk,kl->ijl", features, attention_matrix)
        # [batch size, embedding size 2, 1]
        controlling_vector_expand = tf.expand_dims(controlling_vector, -1)        
        # [batch size, length, 1]
        temp = tf.matmul(temp, controlling_vector_expand)

        # calculate the weight
        # [batch size, length, 1]
        weight_vector = tf.nn.softmax(temp, dim=1)
        # weight sum of features
        # [batch size, embedding size 1, 1]
        weight_sum = tf.matmul(features, weight_vector, transpose_a=True)
        weight_sum = tf.reduce_max(weight_sum, axis=2)            
        return weight_sum
    
    ''' 
    @tensor : [batch size, channel size, tree length, embedding size]
    @return : [batch size, tree length, embedding size]
    '''
    def __multi_channel_pooling(self, tensor):
        height = int(tensor.get_shape()[1])
        result = tf.nn.max_pool(tensor, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        result = tf.reduce_max(result, axis=1)
        return result
