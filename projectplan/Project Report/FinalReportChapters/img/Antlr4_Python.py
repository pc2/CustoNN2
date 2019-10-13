import sys
import os
import re
import pathlib
import argparse
from antlr4 import *
import antlr4
from CLexer import CLexer
from CListener import CListener
from CParser import CParser

class CPrintListener(CListener):
    def __init__(self, filename, verbosity_needed):
        self.filename = filename
        self.verbosity_needed = verbosity_needed

    # variables to calculate number of operations
    val_start = 0
    val_until = 0
    val_diff_func = [1] * 50
    val_diff_inside = [1] * 50

    # value of stride. By default it is set to 1.
    val_stride = 1
    # to store number of for loops in a function
    for_loop_count_local = 0

    # flags to identify position in grammar tree
    inside_function = 0
    inside_for_declaration = 0
    inside_for_condition = 0
    inside_for_expession_no = 0

    loop_block = 0
    iteration_no = 0
    current_loop = 0
    current_func = 0

    # flag to identify if we have entered directdeclarator twice
    # func name resides in this position

    entertwice = 0
    # list to store names of functions we encounter

    func_name = []
    def enterStatement(self, ctx):
        # reset flags to say we are inside a statement
        self.inside_for_declaration = 0
        self.inside_for_condition = 0

    def enterPrimaryExpression(self, ctx):
        myEpression = ctx.getText()
        if self.inside_for_declaration == 1:
            if myEpression.isdigit():
                myEpression = int(myEpression)
                self.val_start = myEpression
          

        if self.inside_for_condition == 1:
            if self.inside_for_expession_no == 1:
                if myEpression.isdigit():
                    myEpression = int(myEpression)
                    self.val_until = myEpression

            if self.inside_for_expession_no == 2:
                if myEpression.isdigit():
                    myEpression = int(myEpression)
                    self.val_stride = myEpression
                    self.inside_for_expession_no = 0

    def enterFunctionDefinition(self, ctx):
        self.inside_function = 1
        self.current_func = self.current_func + 1

    def enterDirectDeclarator(self, ctx):
        # a way to uniquely identify function names
        # from grun , we know that func names are 2 nodes deep after entering a function
        # this methof makes sure that we extract only func names and not other identifiers

        self.entertwice = self.entertwice + 1
        if self.entertwice == 2 and self.inside_function == 1:
            self.func_name.append(ctx.getText())
            self.entertwice = 0

    def exitDirectDeclarator(self, ctx):
        self.entertwice = 0

    def exitFunctionDefinition(self, ctx):

        self.val_diff_func[self.current_func] = \
            sum(self.val_diff_inside[1:self.loop_block + 1])
        self.val_diff_inside = [1] * 50
        self.loop_block = 0
        self.entertwice = 0
        self.inside_function = 0

    def enterForCondition(self, ctx):
        pass

    def exitForCondition(self, ctx):
        self.val_diff_inside[self.loop_block] = \
            self.val_diff_inside[self.loop_block] * (abs(self.val_until
                - self.val_start) / self.val_stride)
        # print("Val of loop_block is {}".format(self.loop_block))
        # print("Val of val diff for loop block is {}".format(self.val_diff_inside[self.loop_block] ))
        self.val_stride = 1

    def enterIterationStatement(self, ctx):
        self.iteration_no += 1
        if self.iteration_no == 1:
            self.loop_block += 1

    def exitIterationStatement(self, ctx):
        self.iteration_no -= 1


    def enterForDeclaration(self, ctx):
        self.inside_for_declaration = 1
        self.inside_for_condition = 0
        self.inside_for_expession_no = 0

    def enterForExpression(self, ctx):
        self.inside_for_declaration = 0
        self.inside_for_condition = 1
        self.inside_for_expession_no = self.inside_for_expession_no + 1

    def exitCompilationUnit(self, ctx):
        sum_convs = 0
        if self.verbosity_needed:
            print 'Number of funcs is {}'.format(self.current_func)
            print 'funcs are {}'.format(self.func_name)

        if self.verbosity_needed:
            for i in range(1, self.current_func + 1):
                print self.val_diff_func[i]

        sum1 = sum(self.val_diff_func[1:self.current_func + 1])
        print 'Total number of operations are {}'.format(sum1)

        if self.verbosity_needed:
            print 'As per our research , only the following kernels contribute to MAC op calculations'
        for (index, funcs) in enumerate(self.func_name):

                # ignore kernels with name Padding in it as they are not MAC.
                # In our kernel files, we have functions like Padding_xyz_conv. We ignore such kernel functions

            if 'Conv' in funcs and 'Pad' not in funcs:
                if self.verbosity_needed:
                    print str(index) + ' is ' + str(funcs) \
                        + 'with Ops :' + str(self.val_diff_func[index
                            + 1])
                sum_convs = sum_convs + self.val_diff_func[index + 1]
        print 'Total number of MAC ops in {} are  '.format(self.filename) \
            + str(sum_convs)
def main():
    parser = argparse.ArgumentParser(description='Generate performance model part a')
    parser.add_argument('-kernelfile', type=str, nargs=1,
                        help='Input file (OpenCL)', required=True)
    parser.add_argument('-v', action='store_true',
                        help='Verbose mode to print more',
                        required=False)
    args = parser.parse_args()
    verbosity_needed = False
    if args.v:
        verbosity_needed = True
        print 'Verbose mode'
    filename = args.kernelfile[0]
    print filename
    ipt = antlr4.FileStream(args.kernelfile[0])
    lexer = CLexer(ipt)
    stream = antlr4.CommonTokenStream(lexer)
    parser = CParser(stream)
    tree = parser.compilationUnit()
    printer = CPrintListener(filename, verbosity_needed)
    walker = antlr4.ParseTreeWalker()
    walker.walk(printer, tree)
main()