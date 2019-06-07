# -*- coding: utf-8 -*-
"""

"""
import nltk
import os

from setting import Path
from setting import tokenize

class Evaluate:
    def __init__(self, paras_list, predict_train=False):
        if (predict_train):
            self.__read_correct_code = self.__read_correct_code_train
            self.__read_predicted_code = self.__read_predicted_code_train
        self.__paras = paras_list[0]
        self.__read_correct_code()
        self.__read_predicted_code()
        self.__evaluate()
    
    def __read_correct_code(self):
        self.__correct_code = []
        
        if (self.__paras.dataset_path == Path.CONALA_PATH):
            path = Path.CONALA_PATH + 'conala-test.json'
            with open(path, 'r', encoding='utf-8') as f:
                null = 'null'
                test_data = eval(f.read())
            
            for data_unit in test_data:
                code = data_unit['snippet']
                self.__correct_code.append(code)
            return
        
        if (self.__paras.dataset_path == Path.HS_PATH):
            path = Path.HS_PATH + 'test_hs.out'
            with open(path, 'r', encoding='utf-8') as f:
                test_out = f.readlines()
                
            for code in test_out:
                if (code == ''): continue
                code = code.replace('§', '\n')
                code = code.replace('\ ', '')
                self.__correct_code.append(code)
            return
    
    def __read_predicted_code(self):
        path = Path.get_prediction_path(self.__paras)
        self.__predicted_code = []
        for i in range(len(self.__correct_code)):
            if os.path.exists(path + str(i)):
                with open(path + str(i), 'r', encoding='utf-8') as f:
                    code = f.read()
                    code = 'null' if code.strip() == '' else code
                    self.__predicted_code.append(code)
            else:
                self.__predicted_code.append('null')
        
    
    def __evaluate(self):
        self.__bleus = []
        for i in range(len(self.__correct_code)):
            self.__bleus.append(nltk.translate.bleu_score.corpus_bleu(
                    reference_corpus=[[tokenize(self.__correct_code[i])]], 
                    translation_corpus=[tokenize(self.__predicted_code[i])])[0])
            
        mean_bleu = sum(self.__bleus) / len(self.__bleus)
        
        log_path = 'evaluate_log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('start evaluating\n')
            f.write('dataset=' + str(self.__paras.dataset_path))
            f.write(', use_pre_train=' + str(self.__paras.use_pre_train))
            f.write(', use_semantic=' + str(self.__paras.use_semantic_logic_order) + '\n')
            for i in range(len(self.__correct_code)):
                f.write('correct : ' + self.__correct_code[i] + '\n')
                f.write('predict : ' + self.__predicted_code[i] + '\n')
                f.write('bleu    : ' + str(self.__bleus[i]) + '\n\n')
            f.write('mean bleu : ' + str(mean_bleu) + '\n\n')
    
    def __read_correct_code_train(self):
        self.__correct_code = []
        
        if (self.__paras.dataset_path == Path.CONALA_PATH):
            path = Path.CONALA_PATH + 'conala-train.json'
            with open(path, 'r', encoding='utf-8') as f:
                null = 'null'
                test_data = eval(f.read())
            
            for data_unit in test_data:
                code = data_unit['snippet']
                self.__correct_code.append(code)
            return
        
        if (self.__paras.dataset_path == Path.HS_PATH):
            path = Path.HS_PATH + 'train_hs.out'
            with open(path, 'r', encoding='utf-8') as f:
                test_out = f.readlines()
                
            for code in test_out:
                if (code == ''): continue
                code = code.replace('§', '\n')
                code = code.replace('\ ', '')
                self.__correct_code.append(code)
            return
    
    def __read_predicted_code_train(self):
        path = Path.get_prediction_path(self.__paras) + '_train/'
        self.__predicted_code = []
        for i in range(len(self.__correct_code)):
            if os.path.exists(path + str(i)):
                with open(path + str(i), 'r', encoding='utf-8') as f:
                    code = f.read()
                    code = 'null' if code.strip() == '' else code
                    self.__predicted_code.append(code)
            else:
                self.__predicted_code.append('null')    
    

if (__name__ == '__main__'):
    from setting import Parameters
    import sys
    handle = Evaluate(Parameters.get_paras_list_from_argv(sys.argv))

#    handle = Evaluate(Parameters.get_paras_list_from_argv(['-h', '-s']), predict_train=True)
