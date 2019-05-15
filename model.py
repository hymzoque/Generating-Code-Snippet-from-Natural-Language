# -*- coding: utf-8 -*-
"""

todo multi channel & attention & cross entropy & adamw

"""
import tensorflow as tf
import setting as setting

'''
    
'''
class Model:
    
    def __init__(self):
        self.__place_alloc()
        self.__embedding()
        self.__bulid_graph()    
    
    '''
    
    '''    
    def __place_alloc(self):
        # None=batch_size
        # descriptions 
        self.input_NL = tf.placeholder(tf.int32, shape=[None, setting.NL_len])
        # nodes of predicted ast
        self.input_ast_nodes = tf.placeholder(tf.int32, shape=[None, setting.Tree_len])
        self.input_ast_parent_nodes = tf.placeholder(tf.int32, shape=[None, setting.Tree_len])
        self.input_ast_grandparent_nods = tf.placeholder(tf.int32, shape=[None, setting.Tree_len])
        # nodes of predicted semantic units
        self.input_semantic_units = tf.placeholder(tf.int32, shape=[None, setting.Semantic_Units_len])
        self.input_children_of_semantic_units = tf.placeholder(tf.int32, shape=[None, setting.Semantic_Unit_len, setting.Semantic_Unit_children_num])
        # indexes of correct output
        self.correct_output = tf.placeholder(tf.float32, shape=[None, setting.tree_node_embedding_size])
        # keep_prob = 1 - dropout
        self.keep_prob = tf.placeholder(tf.float32)
    '''
    
    '''   
    def __embedding(self):
        self.vocabulary_embedding = tf.get_variable('vocabulary_embedding', shape=[setting.vocabulary_num, setting.vocabulary_embedding_size], initializer=self.__initializer())
        self.tree_node_embedding = tf.get_variable('tree_node_embedding', shape=[setting.tree_node_num, setting.tree_node_embedding_size], initializer=self.__initializer())
#        if (use_pre_train):
#            # todo
#            self.vocabulary_embedding_pre_trained
#            self.tree_node_embedding_pre_trained
        # tf if
        
    '''
    
    '''
    def __bulid_graph(self):
        '''
        input of description
        '''
        nl_embedding = tf.nn.embedding_lookup(self.vocabulary_embedding, self.input_NL)
        nl_features = self.__deep_CNN(nl_embedding, setting.vocabulary_embedding_size)
        
        ''' 
        input of parent relation tree nodes
        '''
        # [batch size, tree length, embedding size]
        ast_nodes_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_ast_nodes)
        ast_parent_nodes_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_ast_parent_nodes)
        ast_grandparent_nodes_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_ast_grandparent_nods)
        # [batch size, tree length, 3, embedding size]  (batch, height, width, channels) -- 'channel last'
        parent_stack = tf.stack([ast_nodes_embedding, ast_parent_nodes_embedding, ast_grandparent_nodes_embedding], 2)
        # [batch size, tree length, 1, embedding size]
        temp = tf.layers.conv2d(parent_stack, setting.tree_node_embedding_size, [1, 3])
        # [batch size, tree length, embedding size]
        temp = tf.reduce_max(temp, axis=2)
        temp = tf.nn.relu(temp)
        # [batch size, tree length, embedding size]
        tree_parent_features = self.__deep_CNN(temp, setting.tree_node_embedding_size)
        
        ''' 
        input of semantic order tree nods 
        '''
        semantic_units_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, self.input_semantic_units)
        semantic_em_list = [semantic_units_embedding]
        # loop
        count = tf.constant(0)
        cond = lambda i, l : tf.less(i, setting.Semantic_Unit_children_num)
        #loop body of semantic embedding processing
        def __loop_body_embedding_processing(self, i, l):
            child = self.input_children_of_semantic_units[:, :, i]
            child_embedding = tf.nn.embedding_lookup(self.tree_node_embedding, child)
            l.append(child_embedding)
            i = tf.add(i, 1)
            return i, l
        body = __loop_body_embedding_processing
        tf.while_loop(cond, body, [count, semantic_em_list])
        #
        semantic_unit_stack = tf.stack(semantic_em_list, 2)
        temp = tf.layers.conv2d(semantic_unit_stack, setting.tree_node_embedding_size, [1, setting.Semantic_Unit_children_num + 1])
        temp = tf.reduce_max(temp, axis=2)
        temp = tf.nn.relu(temp) 
        semantic_order_features = self.__deep_CNN(temp, setting.tree_node_embedding_size)
        
        '''
        attention
        
        '''
        # [batch size, nl/tree embedding size]
        nl_features_pooling = self.__max_height_pooling(nl_features)
        tree_parent_features_pooling = self.__max_height_pooling(tree_parent_features)
        semantic_order_features_pooling = self.__max_height_pooling(semantic_order_features)
        
        attention_result_nl = self.__attention(nl_features, semantic_order_features_pooling)
        attention_result_tree_parent = self.__attention(tree_parent_features, nl_features_pooling)
        attention_result_semantic_order = self.__attention(semantic_order_features, nl_features_pooling)
        
        full_connect_input = tf.concat([
                nl_features_pooling,
                tree_parent_features_pooling,
                semantic_order_features_pooling,
                attention_result_nl,
                attention_result_tree_parent,
                attention_result_semantic_order], axis = 1)
        
        ''' 
        full connect with single hidden layer
        '''
        w1 = tf.get_variable('hidden_fc_w1', shape=[int(full_connect_input.shape[1]), setting.hidden_layer_width], initializer=self.__initializer())
        b1 = tf.get_variable('hidden_fc_b1', shape=[setting.hidden_layer_width], initializer=self.__initializer())
        w2 = tf.get_variable('hidden_fc_w2', shape=[setting.hidden_layer_width, setting.tree_node_num], initializer=self.__initializer())
        b2 = tf.get_variable('hidden_fc_b2', shape=[setting.tree_node_num], initializer=self.__initializer())
        temp = tf.nn.tanh(tf.matmul(full_connect_input, w1) + b1)
        temp = tf.nn.dropout(temp, self.keep_prob)
        # logits
        logits = tf.matmul(temp, w2) + b2
        # softmax output   [batch size, tree node num]
        self.predict_output = tf.nn.softmax(logits)
        
        ''' 
        output & optimize
        '''
        # [batch size]
        predict_result = tf.equal(tf.argmax(self.predict_output, 1), tf.argmax(self.correct_output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(predict_result, tf.float32))
        # [batch size]
        batch_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.correct_output, logits=logits)
        self.cross_entropy = tf.reduce_mean(batch_cross_entropy)
        self.optimize = tf.contrib.opt.AdamWOptimizer(weight_decay=setting.weight_decay, learning_rate=setting.learning_rate).minimize(self.cross_entropy)



    ''' weight initializer '''
    def __initializer(self):
        return tf.truncated_normal_initializer(stddev=0.1)        
    
    '''
    resnet deep CNN
    '''    
    def __deep_CNN(self, tensor, channel_size):
        loop_time = tf.constant(int(setting.cnn_deepth / 2))
        count = tf.constant(0)
        cond = lambda i, t, c : tf.less(i, loop_time)       
        def __loop_body_deep_CNN(i, t, channel_size):
            temp = tf.layers.conv1d(t, channel_size , setting.deep_CNN_kernel_size, padding='same')
            temp = tf.nn.relu(temp)
            temp = tf.layers.conv1d(temp, channel_size , setting.deep_CNN_kernel_size, padding='same')
            temp = tf.add_n([temp, t])
            temp = tf.nn.relu(temp)
            i = tf.add(i, 1)
            return i, temp, channel_size
        body = __loop_body_deep_CNN
        count, tensor, c = tf.while_loop(cond, body, [count, tensor, channel_size])
        return tensor
    
    '''
    @tensor : [batch_size, tree/nl_length(height), tree/nl_embedding_size(width)]
    @return : [batch_size, tree/nl_embedding_size]
    '''
    def __max_height_pooling(self, tensor):
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
    
    

    
    
    
    
    
    
    
    
    
    
