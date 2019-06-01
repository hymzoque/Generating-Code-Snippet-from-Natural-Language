# -*- coding: utf-8 -*-
"""

"""
class Grammar:
    def __init__(self):
        import astunparse
        astunparse.Unparser._Store = Grammar.stub
        astunparse.Unparser._Load = Grammar.stub
        astunparse.Unparser._NoneType = Grammar.stub
    
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
            n = '_' + parent.replace('.', '_')
            if not (hasattr(self, n)):
                return True
            meth = getattr(self, n)
            return meth(child, position)
        
        # won't predict <Empty_Node>
        if (child == '<Empty_Node>'):
            return False
        
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
        if position == 0 and not child == 'ast.arguments':
            return False
        if position in [1,2] and not self.__is_list(child):
            return False
        return True
    
    ''' Call('func', 'args', 'keywords'), args and keywords must be <List> '''
    def _ast_Call(self, child, position):
        return False if ((position == 0 or position == 1) and not (self.__is_list(child))) else True
    
    def _ast_Module(self, child, position):
        return True if self.__is_list(child) else False
    
    def _ast_Interactive(self, child, position):
        return True if self.__is_list(child) else False    
    
    def _ast_ImportFrom(self, child, position):
        return False if (position == 0 and not self.__is_list(child)) else True    
    
    def _ast_Assign(self, child, position):
        return False if (position == -1 and not self.__is_list(child)) else True
    
    def _ast_Try(self, child, position):
        return False if (position == 0 and not self.__is_list(child)) else True
    
    def _ast_ListComp(self, child, position):
        return False if (position == 0 and not self.__is_list(child)) else True
    
    def _ast_GeneratorExp(self, child, position):
        return False if (position == 0 and not self.__is_list(child)) else True
    
    def _ast_SetComp(self, child, position):
        return False if (position == 0 and not self.__is_list(child)) else True

    def _ast_DictComp(self, child, position):
        return False if (position == 1 and not self.__is_list(child)) else True
    
    def _ast_comprehension(self, child, position):
        return False if (position == 1 and not self.__is_list(child)) else True

    def _ast_Compare(self, child, position):
        if ((position == 0 or position == 1) and not self.__is_list(child)):
            return False
        # 
        return True
    
    def _ast_arguments(self, child, position):
        if (position in [-1, 1, 2, 4] and not self.__is_list(child)):
            return False
        if (position in [0, 3] and not (child == 'ast.arg' or child == '<None_Node>')):
            return False
        return True
    
    def _ast_arg(self, child, position):
        return False if (position == -1 and not self.__is_str(child)) else True
    
    def _ast_Import(self, child, position):
        return False if not self.__is_list(child) else True

    def _ast_Delete(self, child, position):
        return False if not self.__is_list(child) else True    

    def _ast_Global(self, child, position):
        return False if not self.__is_list(child) else True  

    def _ast_Nonlocal(self, child, position):
        return False if not self.__is_list(child) else True  
    
    def _ast_With(self, child, position):
        return False if (position == -1 and not self.__is_list(child)) else True
    
    def _ast_List(self, child, position):
        return False if (position == -1 and not self.__is_list(child)) else True

    def _ast_Set(self, child, position):
        return False if not self.__is_list(child) else True  

    def _ast_Dict(self, child, position):
        return False if not self.__is_list(child) else True  
    
    def _ast_Tuple(self, child, position):
        return False if (position == -1 and not self.__is_list(child)) else True
    
    def _ast_BoolOp(self, child, position):
        if (position == 0 and not self.__is_list(child)):
            return False 
        if (position == -1 and not (child == 'ast.And' or child == 'ast.Or')):
            return False
        return True

    def _ast_ExtSlice(self, child, position):
        return False if not self.__is_list(child) else True  
    
    def _ast_Str(self, child, position):
        return False if (position == -1 and not self.__is_str(child)) else True
    
    def _ast_Name(self, child, position):
        return False if (position == -1 and not self.__is_str(child)) else True
    
    def _ast_Num(self, child, position):
        return False if (position == -1 and not self.__is_int(child) and not self.__is_float(child)) else True                
    
    def _ast_Subscript(self, child, position):
        return False if (position == 0 and not (child == 'ast.Index' or child =='ast.Slice')) else True     
    
    def _ast_BinOp(self, child, position):
        if ((position == 0) and not 
            (child == 'ast.Add' or child == 'ast.Sub' or child == 'ast.Mult' or child == 'ast.Div' or child == 'ast.Mod' or 
             child == 'ast.LShift' or child == 'ast.RShift' or child == 'ast.BitOr' or child == 'ast.BitXor' or child == 'ast.BitAnd' or 
             child == 'ast.FloorDiv' or child == 'ast.Pow' or child == 'ast.MatMult')):
            return False
        return True
    
    def _ast_UnaryOp(self, child, position):
        if ((position == -1) and not 
            (child == 'ast.Invert' or child == 'ast.Not' or child == 'ast.UAdd' or child == 'ast.USub')):
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
    
