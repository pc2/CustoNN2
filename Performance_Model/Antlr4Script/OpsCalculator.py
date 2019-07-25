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
    val_diff_func = [1] * 50
    
    val_diff_inside = [1] * 50
    
     
    val_stride = 1

    for_loop_count_local = 0
    
    inside_function = 0 
    inside_for_declaration = 0
    inside_for_condition = 0
  
    inside_for_expession_no = 0 

    loop_block = 0
    entertwice = 0
    
    func_name = []

    iteration_no = 0
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
 				#print("Val of val stride is {}".format(self.val_stride))
 
                              
    				 
 
        
    def enterFunctionDefinition(self, ctx):
        self.inside_function = 1
        self.current_func = self.current_func + 1  
        


 
    def enterDirectDeclarator(self, ctx):         
        self.entertwice = self.entertwice + 1
        if self.entertwice == 2 and self.inside_function == 1:
            self.func_name.append(ctx.getText())              
	    self.entertwice = 0

    def exitDirectDeclarator(self, ctx):         
        self.entertwice = 0



    def exitFunctionDefinition(self, ctx):
 
	self.val_diff_func[self.current_func] =  sum(self.val_diff_inside[1:self.loop_block +1])
 	#print("Current func is {}".format(self.current_func))
	 
        self.val_diff_inside = [1] * 50
        self.loop_block = 0
        self.entertwice = 0
        self.inside_function = 0

    def enterForCondition(self, ctx):	 
	pass

    def exitForCondition(self, ctx): 
	self.val_diff_inside[self.loop_block]  = self.val_diff_inside[self.loop_block] * (abs(self.val_until - self.val_start)/self.val_stride)
        #print("Val of loop_block is {}".format(self.loop_block))
	#print("Val of val diff for loop block is {}".format(self.val_diff_inside[self.loop_block] ))

	self.val_stride = 1
	
            
    
    def enterIterationStatement(self, ctx):
        self.iteration_no +=1 
        if (self.iteration_no == 1) :
		self.loop_block += 1

    
    def exitIterationStatement(self, ctx):
	self.iteration_no -=1
        #if (self.iteration_no == 0):
		#self.loop_block -= 1
        


				 
		          
 	

	
    
    def enterForDeclaration(self, ctx):
        self.inside_for_declaration = 1
        self.inside_for_condition  = 0 
        self.inside_for_expession_no = 0
 
    def enterForExpression(self, ctx):
        self.inside_for_declaration = 0
        self.inside_for_condition  = 1 
	self.inside_for_expession_no = self.inside_for_expession_no + 1
 
     
    def exitCompilationUnit(self, ctx):
         
        sum_convs = 0
	print("Number of funcs is {}".format(self.current_func))
        print("funcs are {}".format(self.func_name))

	for i in range (1 , self.current_func+1):
		print(self.val_diff_func[i])
	
	sum1 = sum(self.val_diff_func[1:self.current_func + 1])
	print("Total number of operations are {}".format(sum1))
        
        for index,funcs in enumerate(self.func_name) :
        	if "Conv" in funcs and "Pad" not in funcs:
			print (str(index) +" is "+ str(funcs) + "with Ops :" + str(self.val_diff_func[index+1]))
                        sum_convs = sum_convs + self.val_diff_func[index+1]

        print("Total number of MAC ops are " +str(sum_convs))


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