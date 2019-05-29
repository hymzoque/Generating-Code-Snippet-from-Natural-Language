# -*- coding: utf-8 -*-
"""

"""
class Grammar:
    def __init__():
        import astunparse
        astunparse._Store = Grammar.stub
        astunparse._Load = Grammar.stub
    
    @staticmethod
    def stub(s, t):
        return
    
    '''
    check grammar by parent, child and position of previous child(position of this child - 1)
    may need to extend
    '''
    def grammar_no_problem(self, parent, child, position):
        # <List> as parent
        # <List> can only have ast. as child
        if (parent == '<List>' and 'ast.' not in child):
            return False
        
        # ast Node as parent
        if ('ast.' in parent):
            meth = getattr(self, '_' + parent.replace('.', '_'))
            return meth(child, position)
        
        # Name and Str must have string as first child(not strict)
        
            
        return True
    
    ''' ClassDef('name', 'bases', 'keywords', 'body', 'decorator_list'), name must be str, else must be <List> '''
    def _ast_ClassDef(self, child, position):
        if position == -1 and not self.__is_str(child):
            return False
        if position in [0,1,2,3] and not self.__is_list(child):
            return False
        return True
    
    ''' FunctionDef('name', 'args', 'body', 'decorator_list', 'returns'), name must be str, args,body,decorator_list must be <List> '''    
    def _ast_FunctionDef(self, child, position):
        if position == -1 and not self.__is_str(child):
            return False
        if position in [0,1,2] and not self.__is_list(child):
            return False
        return True
    
    ''' Call('func', 'args', 'keywords'), args and keywords must be <List> '''
    def _ast_Call(self, child, position):
        if (position == 0 or position == 1):
            if not (self.__is_list(child)):
                return False
        return True
    
    def _ast_Module(self, child, position):
        return True if self.__is_list(child) else False
    
    def _ast_Assign(self, child, position):
        if (position == -1 and not self.__is_list(child)):
            return False
        return True
    
    def _ast_Try(self, child, position):
        if (position == 0 and not self.__is_list(child)):
            return False
        return True
    
    def _ast_Str(self, child, position):
        if (position == -1 and not self.__is_str(child)):
            return False
        return True
    
    def _ast_Name(self, child, position):
        if (position == -1 and not self.__is_str(child)):
            return False
        return True
    
    def _ast_Num(self, child, position):
        if (position == -1 and not self.__is_int(child) and not self.__is_float(child)):
            return False
        return True                
    
    def __is_list(self, node):
        return node == '<List>' or node == '<Empty_List>'
    
    def __is_str(self, node):
        return ('ast.' not in node) and (not self.__is_list(node)) and (not self.__is_int(node)) and (not self.__is_float(node)) and (not self.__is_bool(node)) and (not self.__is_None(node))
    
    def __is_int(self, node):
        return node.isdigit()
    def __is_float(self, node):
        try:
            float(node)
            return True
        except:
            return False
    def __is_bool(self, node):
        return node == 'True' or node == 'False'
    def __is_None(self, node):
        return node == '<None_Node>'
    