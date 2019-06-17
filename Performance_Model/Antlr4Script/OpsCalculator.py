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
 
 
   
    val_start = 0
    val_until = 0
    val_diff = [1] * 50
     
    val_stride = 1
    
    inside_for_declaration = 0
    inside_for_condition = 0
  
    inside_for_expession_no = 0 
    
    current_loop = 0
    current_func = 0

    def enterStatement(self, ctx):
        self.inside_for_declaration = 0
        self.inside_for_condition  = 0 
         

 
    def enterPrimaryExpression(self, ctx):
	myEpression = ctx.getText()
         
        if(self.inside_for_declaration == 1 ) :
		if myEpression.isdigit() :
			myEpression = int(myEpression)
			self.val_start = myEpression
			#print("Val of val start is {}".format(self.val_start))

        if(self.inside_for_condition == 1 ) :
		if(self.inside_for_expession_no == 1 ) :
	        	if myEpression.isdigit() :
				myEpression = int(myEpression)
				self.val_until = myEpression 
				#print("Val of val until is {}".format(self.val_until))
				 
		if(self.inside_for_expession_no == 2 ) :
	        	if myEpression.isdigit() :
				myEpression = int(myEpression)
				self.val_stride = myEpression                  
                 		self.inside_for_expession_no = 0
 
 
                              
    				 

 
 
    def enterIterationStatement(self, ctx):
 	pass
        
    def enterFunctionDefinition(self, ctx):
        self.current_func = self.current_func + 1        
	self.val_stride = 1
 
    def exitForCondition(self, ctx): 
	self.val_diff[self.current_func]  = self.val_diff[self.current_func] * ((self.val_until - self.val_start)/self.val_stride)	 
	self.val_stride = 1 
	
            
 	

	
    
    def enterForDeclaration(self, ctx):
        self.inside_for_declaration = 1
        self.inside_for_condition  = 0 
        self.inside_for_expession_no = 0
 
    def enterForExpression(self, ctx):
        self.inside_for_declaration = 0
        self.inside_for_condition  = 1 
	self.inside_for_expession_no = self.inside_for_expession_no + 1
 
     
    def exitCompilationUnit(self, ctx):
	print("Number of funcs is {}".format(self.current_func))
	for i in range (1 , self.current_func + 1):
		print(self.val_diff[i])	
	sum1 = sum(self.val_diff[1:self.current_func + 1])
	print("Total number of operations are {}".format(sum1))


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