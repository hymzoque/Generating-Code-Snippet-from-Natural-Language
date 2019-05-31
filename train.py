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
        
        if (paras.test):
            self.train = self.test_train
        
    ''' test the training and valid time of one batch '''
    def test_train(self):
        data_handle = data.Data(self.__paras)
        nn_model = model.Model(self.__paras)
        with tf.Session(config=self.__gpu_config()) as sess:
            self.__get_ckpt(sess, nn_model)
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
        log.write('dataset=' + str(self.__paras.dataset_path))
        log.write(', use_pre_train=' + str(self.__paras.use_pre_train))
        log.write(', use_semantic=' + str(self.__paras.use_semantic_logic_order) + '\n')
        log.write('training for ' + str(self.__paras.train_times) + ' times\n')
        

        
        with tf.Session(config=self.__gpu_config()) as sess:
            # model file
            self.__get_ckpt(sess, nn_model)
            # summary writer
            summary_path = Path.get_summary_path(self.__paras)
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            train_writer = tf.summary.FileWriter(summary_path + 'train', sess.graph)
            test_writer = tf.summary.FileWriter(summary_path + 'test', sess.graph)
            # train loop
            best_accuracy = 0
            for train_loop in range(self.__paras.train_times):
                start_time = time.time()
                train_summary = self.__train_once(sess, data_handle, nn_model)
                train_writer.add_summary(train_summary, train_loop)
                
                valid_accuracy, test_summary = self.__valid(sess, data_handle, nn_model)
                test_writer.add_summary(test_summary, train_loop)
                end_time = time.time()
                
                log.write('epoch ' + str(train_loop + 1) + ' :\n')
                log.write('accuracy is : ' + str(valid_accuracy) + '\n')
                log.write('    time used : ' + str(end_time - start_time) + '\n')
                # save model if accuracy get better
                if (valid_accuracy > best_accuracy):
                    self.__save_ckpt(sess)
                    best_accuracy = valid_accuracy
                    log.write('better result found\n')
                log.write('\n')
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
            _, summary = session.run(
                    [model.optimize, model.merged],
                    feed_dict={
                            model.input_NL : train_batches[count][0],
                            model.input_ast_nodes : train_batches[count][1],
                            model.input_ast_parent_nodes : train_batches[count][2],
                            model.input_ast_grandparent_nodes : train_batches[count][3],
                            model.input_semantic_units : train_batches[count][4],
                            model.input_children_of_semantic_units : train_batches[count][5],
                            model.correct_output : train_batches[count][6],
                            model.keep_prob : self.__paras.keep_prob})
        return summary
    
    ''' '''
    def __valid(self, session, data_handle, model):
        valid_batches = data_handle.get_valid_batches()
        batch_num = len(valid_batches)
        
        accuracy = 0
        for count in range(batch_num):
            batch_accuracy, summary = session.run(
                    [model.accuracy, model.merged],
                    feed_dict={
                            model.input_NL : valid_batches[count][0],
                            model.input_ast_nodes : valid_batches[count][1],
                            model.input_ast_parent_nodes : valid_batches[count][2],
                            model.input_ast_grandparent_nodes : valid_batches[count][3],
                            model.input_semantic_units : valid_batches[count][4],
                            model.input_children_of_semantic_units : valid_batches[count][5],
                            model.correct_output : valid_batches[count][6],
                            model.keep_prob : 1.0})
            accuracy += batch_accuracy
        accuracy /= batch_num
        return accuracy, summary
    
    ''' restore or create new checkpoint '''
    def __get_ckpt(self, session, model):
        dir_path = self.__model_dir
        if (os.path.exists(dir_path + 'checkpoint')):
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(dir_path))
            print('restoring model')
        else:
            session.run(tf.global_variables_initializer())
            print('new model')
            
            # new mode with pre train weight
            if (self.__paras.use_pre_train):
                path = self.__paras.dataset_path + Path.PRE_TRAIN_WEIGHT_PATH
                if not (os.path.exists(path + 'checkpoint')):
                    raise Exception('have not done the pre train')
                saver = tf.train.Saver([model.pre_train_tree_node_embedding])
                saver.restore(session, tf.train.latest_checkpoint(path))
                print('restoring pre train weight')
        
    ''' save checkpoint '''
    def __save_ckpt(self, session):
        dir_path = self.__model_dir
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, dir_path + 'model.ckpt')
        
        
        
if (__name__ == '__main__'):
    tf.reset_default_graph()
    from setting import Parameters
    import sys
    handle = Train(Parameters.get_paras_from_argv(sys.argv))
    handle.train()
