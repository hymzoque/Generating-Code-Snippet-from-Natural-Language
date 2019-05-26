# -*- coding: utf-8 -*-
"""


"""
import tensorflow as tf
import time
import datetime
import os

import data
import model
from setting import Path

class Train:
    def __init__(self, paras):
        self.__paras = paras
        self.__model_dir = Path.get_model_path(paras)
        
    ''' test the training and valid time of one batch '''
    def test_train(self):
        data_handle = data.Data(self.__paras)
        nn_model = model.Model(self.__paras)
        with tf.Session(config=self.__gpu_config()) as sess:
#            self.__get_ckpt(sess)
            start = time.time()
            
            self.__train_once(sess, data_handle, nn_model)
            mid = time.time()
            print('train time used : ' + str(mid - start))
            
            self.__valid(sess, data_handle, nn_model)
            end = time.time()
            print('valid time used : ' + str(end - mid))
#            self.__save_ckpt(sess)
            
    ''' train method '''
    def train(self):
        data_handle = data.Data(self.__paras)
        nn_model = model.Model(self.__paras)
        log = open('train_log', mode='a')
        log.write(str(datetime.datetime.now()) + '\n')
        log.write('start training\n')
        log.write('training for ' + str(self.__paras.train_times) + ' times\n')
        with tf.Session(config=self.__gpu_config()) as sess:
            # model file
            self.__get_ckpt(sess)
            # train loop
            best_accuracy = 0
            for train_loop in range(self.__paras.train_times):
                start_time = time.time()
                self.__train_once(sess, data_handle, nn_model)
                valid_accuracy = self.__valid(sess, data_handle, nn_model)
                end_time = time.time()
                
                log.write('epoch ' + str(train_loop + 1) + ' :\n')
                log.write('accuracy is : ' + str(valid_accuracy) + '\n')
                log.write('    time used : ' + str(end_time - start_time) + '\n\n')
                # save model if accuracy get better
                if (valid_accuracy > best_accuracy):
                    self.__save_ckpt(sess)
                    best_accuracy = valid_accuracy
                
                log.flush()
        log.write(str(datetime.datetime.now()) + '\n')
        log.write('end training\n\n')
        log.close()        
                
    ''' gpu config '''            
    def __gpu_config(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return config            
                
    ''' train one epoch '''
    def __train_once(self, session, data_handle, model):
        # train
        train_batches = data_handle.get_train_batches()
        batch_num = len(train_batches)
        for count in range(batch_num):
            model.optimize.run(
                    session=session,
                    feed_dict={
                            model.input_NL : train_batches[count][0],
                            model.input_ast_nodes : train_batches[count][1],
                            model.input_ast_parent_nodes : train_batches[count][2],
                            model.input_ast_grandparent_nods : train_batches[count][3],
                            model.input_semantic_units : train_batches[count][4],
                            model.input_children_of_semantic_units : train_batches[count][5],
                            model.correct_output : train_batches[count][6],
                            model.keep_prob : self.__paras.keep_prob,
                            model.pre_train_tree_node_embedding : data.Data.get_pre_train_weight(self.__paras)
                            }
                    )
    
    ''' '''
    def __valid(self, session, data_handle, model):
        valid_batches = data_handle.get_valid_batches()
        batch_num = len(valid_batches)
        
        accuracy = 0
        for count in range(batch_num):
            batch_accuracy = session.run(
                    [model.accuracy],
                    feed_dict={
                            model.input_NL : valid_batches[count][0],
                            model.input_ast_nodes : valid_batches[count][1],
                            model.input_ast_parent_nodes : valid_batches[count][2],
                            model.input_ast_grandparent_nods : valid_batches[count][3],
                            model.input_semantic_units : valid_batches[count][4],
                            model.input_children_of_semantic_units : valid_batches[count][5],
                            model.correct_output : valid_batches[count][6],
                            model.keep_prob : 1.0,
                            model.pre_train_tree_node_embedding : data.Data.get_pre_train_weight(self.__paras)                
                            })
            accuracy += batch_accuracy[0]
        accuracy /= batch_num
        return accuracy
    
    ''' restore or create new checkpoint '''
    def __get_ckpt(self, session):
        dir_path = self.__model_dir
        if (os.path.exists(dir_path + 'checkpoint')):
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(dir_path))
            print('restoring model')
        else:
            session.run(tf.global_variables_initializer())
            print('new model')
        
    ''' save checkpoint '''
    def __save_ckpt(self, session):
        dir_path = self.__model_dir
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, dir_path + 'model.ckpt')
        
        
        
if (__name__ == '__main__'):
    tf.reset_default_graph() # for spyder
    from setting import Parameters
    import sys
    handle = Train(Parameters.get_paras_from_argv(sys.argv))
    #handle.test_train()
    handle.train()
