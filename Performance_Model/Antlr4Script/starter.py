import sys, os, re
import pathlib
import argparse
from antlr4 import *
import antlr4 
from CLexer import CLexer

from CListener import CListener

 

from CParser import CParser

 

import sys




class CPrintListener(CListener):
 
    block_no = 0
    num_func = 0
    expression_list = []
   
    val_start = 0
    val_until = 0
    val_diff = [None] * 10
     
    loop_ops = []
    
    inside_for_declaration = 0
    inside_for_condition = 0
  
    current_loop = 0
    current_block = 0

    def enterStatement(self, ctx):
        self.inside_for_declaration = 0
        self.inside_for_condition  = 0 
         

 
    def enterPrimaryExpression(self, ctx):
	myEpression = ctx.getText()
         
        if(self.inside_for_declaration == 1 ) :
		if myEpression.isdigit() :
			myEpression = int(myEpression)
			self.val_start = myEpression
			print(myEpression)

        if(self.inside_for_condition == 1 ) :
	        if myEpression.isdigit() :
			myEpression = int(myEpression)
			self.val_until = myEpression        


    def exitPrimaryExpression(self, ctx):
        pass 
	 
 
    def enterIterationStatement(self, ctx):
        self.block_no = self.block_no + 1 ;
        print("Entering for loop {}".format(self.block_no))
        
 

 
    def exitIterationStatement(self, ctx):
         
 	 

        print("exiting for loop  {}".format(self.block_no))         
        if (self.block_no  == 1 ) :
		self.num_func = self.num_func + 1
                print("Total number of for funcs {}".format(self.num_func))
                
                
        self.block_no =  self.block_no - 1    
	

 
    def enterForExpression(self, ctx):
        self.inside_for_declaration = 0
        self.inside_for_condition  = 1 
	 
    
    def enterForCondition(self, ctx):
    	self.current_loop = self.current_loop + 1
    	 
    
    def exitForCondition(self, ctx):
	pass    
	
 

    def extract_loop_count(self, node):
        pass
		
 
    
    def enterForDeclaration(self, ctx):
        self.inside_for_declaration = 1
        self.inside_for_condition  = 0 
 

    
    def exitCompilationUnit(self, ctx):
         for i in range (len(self.val_diff) - 1):
		print(self.val_diff[i])


def main():
    
    parser = argparse.ArgumentParser(description='Generate performance model part a')
    parser.add_argument('-kernelfile', type=str, nargs=1,
                        help='Input file (OpenCL)', required=True)
 
 
    args = parser.parse_args()


    ipt = antlr4.FileStream(args.kernelfile[0])

 

    lexer = CLexer(ipt)
    stream = antlr4.CommonTokenStream(lexer)
    parser = CParser(stream)

    tree = parser.compilationUnit()

    printer = CPrintListener()
    walker = antlr4.ParseTreeWalker()
    walker.walk(printer, tree)

 
 

main()
