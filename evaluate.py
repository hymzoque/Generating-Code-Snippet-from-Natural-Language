# -*- coding: utf-8 -*-
"""

"""
import re

import bleu_score
from setting import Path

class Evaluate:
    def __init__(self, paras):
        self.__paras = paras
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
                if (code == 'null'): continue
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
            with open(path + str(i), 'r', encoding='utf-8') as f:
                self.__predicted_code.append(f.read())
        
    
    def __evaluate(self):
        self.__bleus = []
        for i in range(len(self.__correct_code)):
            self.__bleus.append(bleu_score.compute_bleu(
                    reference_corpus=self.__tokenize(self.__correct_code[i]), 
                    translation_corpus=self.__tokenize(self.__predicted_code[i]))[0])
            
        mean_bleu = sum(self.__bleus) / len(self.__bleus)
        
        log_path = 'evaluate_log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n')
            for i in range(len(self.__correct_code)):
                f.write('correct : ' + self.__correct_code[i] + '\n')
                f.write('predict : ' + self.__predicted_code[i] + '\n')
                f.write('bleu    : ' + str(self.__bleus[i]) + '\n\n')
            f.write('mean bleu : ' + str(mean_bleu))
            
    
    '''
    from https://github.com/conala-corpus/conala-baseline/blob/master/eval/conala_eval.py 
    tokenize_for_bleu_eval()
    '''
    def __tokenize(self, code):
        code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]

        return tokens
    
    
if (__name__ == '__main__'):
    from setting import Parameters
    import sys
    handle = Evaluate(Parameters.get_paras_from_argv(sys.argv))