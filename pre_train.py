# -*- coding: utf-8 -*-
"""

"""
import tensorflow as tf
import numpy as np
import os

from setting import Path

'''
use node and grandparent to predict parent

'''
class Pre_train:
    
    def __init__(self, paras):
        self.__paras = paras
        self.__process_pre_train_data()
        self.__pre_train()
    
    '''
    '''
    def __process_pre_train_data(self):
        with open(self.__paras.dataset_path + Path.PRE_TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
            pre_train_data = eval(f.read())
        pre_train_data = np.array(pre_train_data)
        self.__input = pre_train_data[:, [0,2]]
        self.__label = pre_train_data[:, [1]]
        
    def __pre_train(self):
        model = Pre_train.__Model(self.__paras)
        with tf.Session(config=self.__gpu_config()) as sess:
            sess.run(tf.global_variables_initializer())

            train_times = 1000
            start_alpha = 0.025
            stop_alpha = 0.0001
            for i in range(train_times):
                learning_rate = start_alpha * (train_times - i) / train_times + stop_alpha * i / train_times
                _, loss = sess.run([model.optimize, model.loss], feed_dict={
                        model.learning_rate : learning_rate,
                        model.input : self.__input,
                        model.labels : self.__label})
            model.normalize.eval()
            
            # save embedding weight
            path = self.__paras.dataset_path + Path.PRE_TRAIN_WEIGHT_PATH
            if not os.path.exists(path):
                os.makedirs(path)
            saver = tf.train.Saver({'pre_train_tree_node_embedding' : model.pre_train_tree_node_embedding})
            saver.save(sess, path + 'weight.ckpt')
            
    
    def __gpu_config(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return config
    
    '''
    pre train model
    tree-based CBOW
    '''
    class __Model:
        def __init__(self, paras):
            self.input = tf.placeholder(tf.int32, shape=[None, 2])
            self.labels = tf.placeholder(tf.int64, shape=[None, 1])
            self.learning_rate = tf.placeholder(tf.float32)
            self.pre_train_tree_node_embedding = tf.get_variable('pre_train_embedding', shape=[paras.tree_node_num, paras.tree_node_embedding_size])

            embed = tf.nn.embedding_lookup(self.pre_train_tree_node_embedding, self.input)
            self.input_embed = tf.reduce_mean(embed, axis=1)

            self.nce_weights = tf.get_variable('nce_weights', shape=[paras.tree_node_num, paras.tree_node_embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.nce_biases = tf.get_variable('nce_biases', shape=[paras.tree_node_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            self.sampler = tf.nn.uniform_candidate_sampler(
                  true_classes=self.labels,
                  num_true=1,
                  num_sampled=64,
                  unique=True,
                  range_max=paras.tree_node_num)
            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                    weights=self.nce_weights,
                    biases=self.nce_biases,
                    labels=self.labels,
                    inputs=self.input_embed,
                    num_sampled=64,
                    num_classes=paras.tree_node_num,
                    sampled_values=self.sampler))
#            tf.summary.scalar('loss_pre_train', self.loss)
            self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            
            # normalize
            embedding = self.pre_train_tree_node_embedding
            normalized_embedding = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), axis=1, keep_dims=True))
            self.normalize = self.pre_train_tree_node_embedding.assign(normalized_embedding)
            
if (__name__ == '__main__'):
    import setting
    tf.reset_default_graph()
    handle = Pre_train(setting.Parameters_conala_base())
    tf.reset_default_graph()
    handle = Pre_train(setting.Parameters_hs_base())

    
