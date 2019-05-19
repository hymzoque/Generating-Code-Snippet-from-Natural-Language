# -*- coding: utf-8 -*-
"""
todo multi dataset
"""
import re

import bleu_score
import setting

class Evaluate:
    def __init__(self):
        self.__read_correct_code_conala()
        self.__read_predicted_code()
        self.__evaluate()
    
    def __read_correct_code_conala(self):
        path = setting.CONALA_PATH + 'conala-test.json'
        with open(path, 'r') as f:
            null = 'null'
            test_data = eval(f.read())
        
        self.__correct_code = []
        for data_unit in test_data:
            code = data_unit['snippet']
            if (code == 'null') : continue
            self.__correct_code.append(code)
        
    
    def __read_predicted_code(self):
        path = 'prediction/'
        self.__predicted_code = []
        for i in range(len(self.__correct_code)):
            with open(path + str(i), 'r') as f:
                self.__predicted_code.append(f.read())
        
    
    def __evaluate(self):
        self.__bleus = []
        for i in range(len(self.__correct_code)):
            self.__bleus.append(bleu_score.compute_bleu(
                    reference_corpus=self.__tokenize(self.__correct_code[i]), 
                    translation_corpus=self.__tokenize(self.__predicted_code[i]))[0])
            
        mean_bleu = sum(self.__bleus) / len(self.__bleus)
        
        log_path = 'evaluate_log'
        with open(log_path, 'w') as f:
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
    
    

handle = Evaluate()