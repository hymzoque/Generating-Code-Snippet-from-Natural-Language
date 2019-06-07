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
    def __init__(self, paras_list):
        self.__paras_list = paras_list
        
#        if (paras.test):
#            self.train = self.test_train
        
#    ''' test the training and valid time of one batch '''
#    def test_train(self):
#        data_handle = data.Data(self.__paras)
#        nn_model = model.Model(self.__paras)
#        with tf.Session(config=self.__gpu_config()) as sess:
#            self.__get_ckpt(sess, nn_model)
#            start = time.time()
#            
#            self.__train_once(sess, data_handle, nn_model)
#            mid = time.time()
#            print('train time used : ' + str(mid - start))
#            
#            self.__valid(sess, data_handle, nn_model)
#            end = time.time()
#            print('valid time used : ' + str(end - mid))
##            self.__save_ckpt(sess)
            
    ''' train method '''
    def train(self):
        paras_base = self.__paras_list[0]
        data_handle = data.Data(self.__paras_list)
        model_dir_list = Path.get_model_path_list(paras_base)
        summary_dir_list = Path.get_summary_path_list(paras_base)
        
        log = open('train_log', mode='a')
        log.write(str(datetime.datetime.now()) + '\n')
        log.write('start training\n')
        log.write('dataset=' + str(paras_base.dataset_path))
        log.write(', use_pre_train=' + str(paras_base.use_pre_train))
        log.write(', use_semantic=' + str(paras_base.use_semantic_logic_order) + '\n')
        
        for paras, model_dir, summary_dir, get_train_batches, get_valid_batches in zip(self.__paras_list, model_dir_list, summary_dir_list, 
                                                                                       data_handle.get_train_batches_methods(), data_handle.get_valid_batches_methods()):
            log.write('training ' + paras.__class__.__name__ + ' for ' + str(paras.train_times) + ' times\n')
            
            with tf.Graph().as_default(), tf.Session(config=self.__gpu_config()) as sess:
                nn_model = model.Model(paras)
                # model file
                self.__get_ckpt(model_dir, paras, sess, nn_model)
                # summary writer
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                train_writer = tf.summary.FileWriter(summary_dir + 'train', sess.graph)
                test_writer = tf.summary.FileWriter(summary_dir + 'test', sess.graph)
                # train loop
                best_accuracy = 0
                for train_loop in range(paras.train_times):
                    start_time = time.time()
                    train_summarys = self.__train_once(paras, sess, get_train_batches, nn_model)
                    
                    batch_num = len(train_summarys)
                    for i in range(batch_num):
                        train_writer.add_summary(train_summarys[i], train_loop * batch_num + i)
                    
                    valid_accuracy, test_summarys = self.__valid(sess, get_valid_batches, nn_model)
                    for i in range(batch_num):
                        if (i == len(test_summarys)):
                            break
                        test_writer.add_summary(test_summarys[i], train_loop * batch_num + i)
                    end_time = time.time()
                    
                    log.write('epoch ' + str(train_loop + 1) + ' :\n')
                    log.write('valid accuracy is : ' + str(valid_accuracy) + '\n')
                    log.write('        time used : ' + str(end_time - start_time) + '\n')
                    # save model if accuracy get better
                    if (valid_accuracy > best_accuracy):
                        best_accuracy = valid_accuracy
                        self.__save_ckpt(model_dir, sess)
                        log.write('better result found\n')
                    log.write('\n')
                    log.flush()            
            log.write('\n')
        log.write(str(datetime.datetime.now()) + '\n')
        log.write('end training\n\n')
        log.close()        
                
    ''' gpu config '''            
    def __gpu_config(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return config            
                
    ''' train one epoch '''
    def __train_once(self, paras, session, get_train_batches, model):
        # train
        train_batches = get_train_batches()
        batch_num = len(train_batches)
        summarys = []
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
                            model.keep_prob : paras.keep_prob
                            #,model.unbalance_weights_table : data.Data.get_unbalance_weights_table(self.__paras)
                            })
            summarys.append(summary)
        return summarys
    
    ''' '''
    def __valid(self, session, get_valid_batches, model):
        valid_batches = get_valid_batches()
        batch_num = len(valid_batches)
        summarys = []
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
                            model.keep_prob : 1.0
                            #,model.unbalance_weights_table : data.Data.get_unbalance_weights_table(self.__paras)
                            })
            summarys.append(summary)
            accuracy += batch_accuracy
            
        accuracy /= batch_num
        return accuracy, summarys
    
    ''' restore or create new checkpoint '''
    def __get_ckpt(self, dir_path, paras, session, model):
        if (os.path.exists(dir_path + 'checkpoint')):
            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(dir_path))
        else:
            session.run(tf.global_variables_initializer())
            # new mode with pre train weight
            if (paras.use_pre_train):
                path = paras.dataset_path + Path.PRE_TRAIN_WEIGHT_PATH
                if not (os.path.exists(path + 'checkpoint')):
                    raise Exception('have not done the pre train')
                saver = tf.train.Saver([model.pre_train_tree_node_embedding])
                saver.restore(session, tf.train.latest_checkpoint(path))
                print('restoring pre train weight')
        
    ''' save checkpoint '''
    def __save_ckpt(self, dir_path, session):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(session, dir_path + 'model.ckpt')
        
        
        
if (__name__ == '__main__'):
    tf.reset_default_graph()
    from setting import Parameters
    import sys
    handle = Train(Parameters.get_paras_list_from_argv(sys.argv))
    handle.train()
